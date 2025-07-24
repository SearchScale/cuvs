/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/preprocessing/quantize/scalar.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_ROWS 50000
#define N_COLS 1024
#define N_QUERIES 8
#define TOPK 50
#define N_CLUSTERS 10

float dataset[N_ROWS][N_COLS];
float queries[N_QUERIES][N_COLS];

typedef struct {
    double float32_build_time, float32_search_time;
    double quant_train_time, quant_transform_time, quant_build_time, quant_search_time;
    float float32_recall, quantized_recall;
} BenchmarkResults;

BenchmarkResults random_results, clustered_results;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void generate_random_dataset() {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N_ROWS; ++i)
        for (int j = 0; j < N_COLS; ++j)
            dataset[i][j] = (float)rand() / RAND_MAX;
}

void generate_clustered_dataset() {
    float cluster_centers[N_CLUSTERS][N_COLS];
    for (int c = 0; c < N_CLUSTERS; ++c)
        for (int j = 0; j < N_COLS; ++j)
            cluster_centers[c][j] = (float)rand() / RAND_MAX;
    
    int points_per_cluster = N_ROWS / N_CLUSTERS;
    float cluster_std = 0.1f;
    for (int i = 0; i < N_ROWS; ++i) {
        int cluster_id = (i / points_per_cluster >= N_CLUSTERS) ? N_CLUSTERS - 1 : i / points_per_cluster;
        for (int j = 0; j < N_COLS; ++j) {
            float u1 = (float)rand() / RAND_MAX, u2 = (float)rand() / RAND_MAX;
            float gaussian = cluster_std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            dataset[i][j] = cluster_centers[cluster_id][j] + gaussian;
            if (dataset[i][j] < 0.0f) dataset[i][j] = 0.0f;
            if (dataset[i][j] > 1.0f) dataset[i][j] = 1.0f;
        }
    }
}

void generate_test_queries() {
    for (int i = 0; i < N_QUERIES; ++i)
        for (int j = 0; j < N_COLS; ++j)
            queries[i][j] = (float)rand() / RAND_MAX;
}

void prepare_tensor(DLManagedTensor* tensor, void* data, DLDeviceType device_type,
                   DLDataTypeCode dtype_code, uint8_t dtype_bits, int64_t* shape, int ndim) {
    tensor->dl_tensor.data = data; tensor->dl_tensor.device.device_type = device_type;
    tensor->dl_tensor.device.device_id = 0; tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.dtype.code = dtype_code; tensor->dl_tensor.dtype.bits = dtype_bits;
    tensor->dl_tensor.dtype.lanes = 1; tensor->dl_tensor.shape = shape;
    tensor->dl_tensor.strides = NULL; tensor->dl_tensor.byte_offset = 0;
    tensor->manager_ctx = NULL; tensor->deleter = NULL;
}

uint32_t** compute_ground_truth() {
    uint32_t **ground_truth = (uint32_t **)malloc(N_QUERIES * sizeof(uint32_t *));
    
    for (int q = 0; q < N_QUERIES; ++q) {
        ground_truth[q] = (uint32_t *)malloc(TOPK * sizeof(uint32_t));
        
        typedef struct {
            float distance;
            uint32_t index;
        } DistanceIndex;
        
        DistanceIndex *all_distances = (DistanceIndex *)malloc(N_ROWS * sizeof(DistanceIndex));
        
        for (int i = 0; i < N_ROWS; ++i) {
            float dist = 0.0f;
            for (int j = 0; j < N_COLS; ++j) {
                float diff = queries[q][j] - dataset[i][j];
                dist += diff * diff;
            }
            all_distances[i].distance = dist; all_distances[i].index = i;
        }
        
        for (int i = 0; i < TOPK; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < N_ROWS; ++j)
                if (all_distances[j].distance < all_distances[min_idx].distance) min_idx = j;
            DistanceIndex temp = all_distances[i];
            all_distances[i] = all_distances[min_idx]; all_distances[min_idx] = temp;
        }
        
        for (int i = 0; i < TOPK; ++i) ground_truth[q][i] = all_distances[i].index;
        
        free(all_distances);
    }
    
    return ground_truth;
}

float compute_recall(uint32_t *search_results, uint32_t *ground_truth) {
    int recall_count = 0;
    for (int k = 0; k < TOPK; ++k) {
        uint32_t result_neighbor = search_results[k];
        for (int gt = 0; gt < TOPK; ++gt)
            if (ground_truth[gt] == result_neighbor) { recall_count++; break; }
    }
    return (float)recall_count / (float)TOPK * 100.0f;
}

void evaluate_and_print_recall(uint32_t *float32_neighbors, uint32_t *quantized_neighbors, 
                              uint32_t **ground_truth, BenchmarkResults* results) {
    printf("\n=== RECALL EVALUATION ===\n");
    float total_float32_recall = 0.0f, total_quantized_recall = 0.0f;
    
    for (int q = 0; q < N_QUERIES; ++q) {
        float float32_recall = compute_recall(&float32_neighbors[q * TOPK], ground_truth[q]);
        float quantized_recall = compute_recall(&quantized_neighbors[q * TOPK], ground_truth[q]);
        
        printf("Query %d - Float32: %.1f%%, Quantized: %.1f%%\n", q, float32_recall, quantized_recall);
        
        total_float32_recall += float32_recall;
        total_quantized_recall += quantized_recall;
    }
    
    float avg_float32_recall = total_float32_recall / N_QUERIES;
    float avg_quantized_recall = total_quantized_recall / N_QUERIES;
    results->float32_recall = avg_float32_recall; results->quantized_recall = avg_quantized_recall;
    
    printf("\n=== SUMMARY ===\n");
    printf("Float32 CAGRA average recall@%d: %.1f%%\n", TOPK, avg_float32_recall);
    printf("Quantized CAGRA average recall@%d: %.1f%%\n", TOPK, avg_quantized_recall);
    printf("Recall degradation due to quantization: %.1f%%\n", avg_float32_recall - avg_quantized_recall);
}

cuvsCagraIndexParams_t create_index_params() {
    cuvsCagraIndexParams_t index_params;
    cuvsCagraIndexParamsCreate(&index_params);
    index_params->build_algo = NN_DESCENT;
    index_params->nn_descent_niter = 50;
    index_params->intermediate_graph_degree = 256;
    index_params->graph_degree = 128;
    index_params->metric = L2Expanded;
    return index_params;
}

cuvsCagraSearchParams_t create_search_params() {
    cuvsCagraSearchParams_t search_params;
    cuvsCagraSearchParamsCreate(&search_params);
    search_params->itopk_size = 2 * TOPK;  // Keep as 2*TOPK as requested
    search_params->search_width = 20;      // Optimal search width for high recall
    search_params->max_iterations = 200;   // More iterations
    search_params->num_random_samplings = 16; // More random sampling
    return search_params;
}

uint32_t* run_float32_cagra_workflow(cuvsResources_t res, BenchmarkResults* results) {
    DLManagedTensor dataset_tensor;
    int64_t dataset_shape[2] = {N_ROWS, N_COLS};
    prepare_tensor(&dataset_tensor, dataset, kDLCPU, kDLFloat, 32, dataset_shape, 2);

    cuvsCagraIndexParams_t index_params = create_index_params();
    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    double start_time = get_time();
    cuvsCagraBuild(res, index_params, &dataset_tensor, index);
    results->float32_build_time = get_time() - start_time;
    printf("Float32 index build time: %.3f seconds\n", results->float32_build_time);

    uint32_t *neighbors;
    float *distances, *queries_d, *dataset_d;
    
    cuvsRMMAlloc(res, (void **)&queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsRMMAlloc(res, (void **)&neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMAlloc(res, (void **)&distances, sizeof(float) * N_QUERIES * TOPK);
    cuvsRMMAlloc(res, (void **)&dataset_d, sizeof(float) * N_ROWS * N_COLS);
    
    cudaMemcpy(dataset_d, dataset, sizeof(float) * N_ROWS * N_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(queries_d, queries, sizeof(float) * N_QUERIES * N_COLS, cudaMemcpyHostToDevice);

    DLManagedTensor queries_tensor, neighbors_tensor, distances_tensor;
    int64_t queries_shape[2] = {N_QUERIES, N_COLS};
    int64_t neighbors_shape[2] = {N_QUERIES, TOPK};
    int64_t distances_shape[2] = {N_QUERIES, TOPK};
    
    prepare_tensor(&queries_tensor, queries_d, kDLCUDA, kDLFloat, 32, queries_shape, 2);
    prepare_tensor(&neighbors_tensor, neighbors, kDLCUDA, kDLUInt, 32, neighbors_shape, 2);
    prepare_tensor(&distances_tensor, distances, kDLCUDA, kDLFloat, 32, distances_shape, 2);

    cuvsCagraSearchParams_t search_params = create_search_params();
    cuvsFilter filter = {NO_FILTER, (uintptr_t)NULL};
    
    start_time = get_time();
    cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter);
    results->float32_search_time = get_time() - start_time;
    printf("Float32 search time: %.3f seconds\n", results->float32_search_time);

    uint32_t *neighbors_h = (uint32_t *)malloc(sizeof(uint32_t) * N_QUERIES * TOPK);
    cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * N_QUERIES * TOPK, cudaMemcpyDefault);

    cuvsCagraSearchParamsDestroy(search_params);
    cuvsRMMFree(res, distances, sizeof(float) * N_QUERIES * TOPK);
    cuvsRMMFree(res, neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMFree(res, dataset_d, sizeof(float) * N_ROWS * N_COLS);
    cuvsRMMFree(res, queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraIndexDestroy(index);

    return neighbors_h;
}

typedef struct {
    cuvsScalarQuantizer_t quantizer;
    int8_t *quantized_dataset;
    int8_t *quantized_queries;
} QuantizationResult;

QuantizationResult quantize_data(cuvsResources_t res, float *dataset_d, float *queries_d, BenchmarkResults* results) {
    DLManagedTensor dataset_d_tensor, queries_tensor;
    int64_t dataset_shape[2] = {N_ROWS, N_COLS};
    int64_t queries_shape[2] = {N_QUERIES, N_COLS};
    prepare_tensor(&dataset_d_tensor, dataset_d, kDLCUDA, kDLFloat, 32, dataset_shape, 2);
    prepare_tensor(&queries_tensor, queries_d, kDLCUDA, kDLFloat, 32, queries_shape, 2);

    cuvsScalarQuantizerParams_t quant_params;
    cuvsScalarQuantizerParamsCreate(&quant_params);
    quant_params->quantile = 0.98;

    cuvsScalarQuantizer_t quantizer;
    cuvsScalarQuantizerCreate(&quantizer);

    double start_time = get_time();
    cuvsScalarQuantizerTrain(res, quant_params, &dataset_d_tensor, quantizer);
    results->quant_train_time = get_time() - start_time;
    printf("Quantizer training time: %.3f seconds\n", results->quant_train_time);

    int8_t *quantized_dataset, *quantized_queries;
    cuvsRMMAlloc(res, (void **)&quantized_dataset, sizeof(int8_t) * N_ROWS * N_COLS);
    cuvsRMMAlloc(res, (void **)&quantized_queries, sizeof(int8_t) * N_QUERIES * N_COLS);

    DLManagedTensor quantized_dataset_tensor, quantized_queries_tensor;
    int64_t quantized_dataset_shape[2] = {N_ROWS, N_COLS};
    int64_t quantized_queries_shape[2] = {N_QUERIES, N_COLS};
    
    prepare_tensor(&quantized_dataset_tensor, quantized_dataset, kDLCUDA, kDLInt, 8, quantized_dataset_shape, 2);
    prepare_tensor(&quantized_queries_tensor, quantized_queries, kDLCUDA, kDLInt, 8, quantized_queries_shape, 2);

    start_time = get_time();
    cuvsScalarQuantizerTransform(res, quantizer, &dataset_d_tensor, &quantized_dataset_tensor);
    cuvsScalarQuantizerTransform(res, quantizer, &queries_tensor, &quantized_queries_tensor);
    results->quant_transform_time = get_time() - start_time;
    printf("Quantization transform time: %.3f seconds\n", results->quant_transform_time);

    cuvsScalarQuantizerParamsDestroy(quant_params);

    QuantizationResult result = {quantizer, quantized_dataset, quantized_queries};
    return result;
}

uint32_t* run_quantized_cagra_workflow(cuvsResources_t res, BenchmarkResults* results) {
    float *queries_d, *dataset_d;
    cuvsRMMAlloc(res, (void **)&queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsRMMAlloc(res, (void **)&dataset_d, sizeof(float) * N_ROWS * N_COLS);
    
    cudaMemcpy(dataset_d, dataset, sizeof(float) * N_ROWS * N_COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(queries_d, queries, sizeof(float) * N_QUERIES * N_COLS, cudaMemcpyHostToDevice);

    QuantizationResult quant_result = quantize_data(res, dataset_d, queries_d, results);
    cuvsScalarQuantizer_t quantizer = quant_result.quantizer;
    int8_t *quantized_dataset = quant_result.quantized_dataset;
    int8_t *quantized_queries = quant_result.quantized_queries;

    DLManagedTensor quantized_dataset_tensor, quantized_queries_tensor;
    int64_t quantized_dataset_shape[2] = {N_ROWS, N_COLS};
    int64_t quantized_queries_shape[2] = {N_QUERIES, N_COLS};
    
    prepare_tensor(&quantized_dataset_tensor, quantized_dataset, kDLCUDA, kDLInt, 8, quantized_dataset_shape, 2);
    prepare_tensor(&quantized_queries_tensor, quantized_queries, kDLCUDA, kDLInt, 8, quantized_queries_shape, 2);

    cuvsCagraIndex_t quant_index;
    cuvsCagraIndexCreate(&quant_index);
    
    cuvsCagraIndexParams_t quant_index_params = create_index_params();
    double start_time = get_time();
    cuvsCagraBuild(res, quant_index_params, &quantized_dataset_tensor, quant_index);
    results->quant_build_time = get_time() - start_time;
    printf("Quantized index build time: %.3f seconds\n", results->quant_build_time);

    uint32_t *quant_neighbors;
    float *quant_distances;
    cuvsRMMAlloc(res, (void **)&quant_neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMAlloc(res, (void **)&quant_distances, sizeof(float) * N_QUERIES * TOPK);

    DLManagedTensor quant_neighbors_tensor, quant_distances_tensor;
    int64_t quant_neighbors_shape[2] = {N_QUERIES, TOPK};
    int64_t quant_distances_shape[2] = {N_QUERIES, TOPK};
    
    prepare_tensor(&quant_neighbors_tensor, quant_neighbors, kDLCUDA, kDLUInt, 32, quant_neighbors_shape, 2);
    prepare_tensor(&quant_distances_tensor, quant_distances, kDLCUDA, kDLFloat, 32, quant_distances_shape, 2);

    cuvsCagraSearchParams_t quant_search_params;
    cuvsCagraSearchParamsCreate(&quant_search_params);
    quant_search_params->itopk_size = 2 * TOPK;  // Always 2 * TOPK
    quant_search_params->search_width = 16;
    quant_search_params->max_iterations = 100;
    quant_search_params->num_random_samplings = 8;
    
    cuvsFilter filter = {NO_FILTER, (uintptr_t)NULL};
    start_time = get_time();
    cuvsCagraSearch(res, quant_search_params, quant_index, &quantized_queries_tensor,
                    &quant_neighbors_tensor, &quant_distances_tensor, filter);
    results->quant_search_time = get_time() - start_time;
    printf("Quantized search time: %.3f seconds\n", results->quant_search_time);

    uint32_t *quant_neighbors_h = (uint32_t *)malloc(sizeof(uint32_t) * N_QUERIES * TOPK);
    cudaMemcpy(quant_neighbors_h, quant_neighbors, sizeof(uint32_t) * N_QUERIES * TOPK, cudaMemcpyDefault);

    cuvsCagraSearchParamsDestroy(quant_search_params);
    cuvsRMMFree(res, quant_distances, sizeof(float) * N_QUERIES * TOPK);
    cuvsRMMFree(res, quant_neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMFree(res, quantized_dataset, sizeof(int8_t) * N_ROWS * N_COLS);
    cuvsRMMFree(res, quantized_queries, sizeof(int8_t) * N_QUERIES * N_COLS);
    cuvsRMMFree(res, dataset_d, sizeof(float) * N_ROWS * N_COLS);
    cuvsRMMFree(res, queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsCagraIndexDestroy(quant_index);
    cuvsScalarQuantizerDestroy(quantizer);
    cuvsCagraIndexParamsDestroy(quant_index_params);

    return quant_neighbors_h;
}

void cagra_build_search_test(const char* dataset_type, void (*dataset_generator)(), BenchmarkResults* results) {
    printf("\n=== %s Dataset ===\n", dataset_type);
    
    dataset_generator();
    generate_test_queries();
    
    cuvsResources_t res;
    cuvsResourcesCreate(&res);

    printf("\n--- Float32 Workflow ---\n");
    uint32_t *neighbors_h = run_float32_cagra_workflow(res, results);
    uint32_t **ground_truth = compute_ground_truth();
    
    printf("\n--- Quantized Workflow ---\n");
    uint32_t *quant_neighbors_h = run_quantized_cagra_workflow(res, results);
    
    evaluate_and_print_recall(neighbors_h, quant_neighbors_h, ground_truth, results);

    free(quant_neighbors_h);
    free(neighbors_h);
    for (int q = 0; q < N_QUERIES; ++q) {
        free(ground_truth[q]);
    }
    free(ground_truth);
    cuvsResourcesDestroy(res);
}

void print_summary_table() {
    printf("\n\n=== PERFORMANCE SUMMARY TABLE ===\n");
    printf("┌───────────┬─────┬─────┬─────┬───────────┬───────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n");
    printf("│ Dataset   │  N  │ Dim │TopK │ F32 Build │ F32 Search│ Q.Train │ Q.Trans │ Q.Build │ Q.Search│ F32 Rec │ Q.Rec   │\n");
    printf("│ Type      │ (k) │     │     │ (sec)     │ (sec)     │ (sec)   │ (sec)   │ (sec)   │ (sec)   │ (%%)     │ (%%)     │\n");
    printf("├───────────┼─────┼─────┼─────┼───────────┼───────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n");
    printf("│ Random    │ %3d │%4d │ %3d │ %9.3f │ %9.3f │ %7.3f │ %7.3f │ %7.3f │ %7.3f │ %7.1f │ %7.1f │\n",
           N_ROWS/1000, N_COLS, TOPK,
           random_results.float32_build_time, random_results.float32_search_time,
           random_results.quant_train_time, random_results.quant_transform_time,
           random_results.quant_build_time, random_results.quant_search_time,
           random_results.float32_recall, random_results.quantized_recall);
    printf("│ Clustered │ %3d │%4d │ %3d │ %9.3f │ %9.3f │ %7.3f │ %7.3f │ %7.3f │ %7.3f │ %7.1f │ %7.1f │\n",
           N_ROWS/1000, N_COLS, TOPK,
           clustered_results.float32_build_time, clustered_results.float32_search_time,
           clustered_results.quant_train_time, clustered_results.quant_transform_time,
           clustered_results.quant_build_time, clustered_results.quant_search_time,
           clustered_results.float32_recall, clustered_results.quantized_recall);
    printf("└───────────┴─────┴─────┴─────┴───────────┴───────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘\n");
    
    printf("\n=== SPEEDUP ANALYSIS ===\n");
    printf("┌───────────┬───────────┬────────────┬─────────────┐\n");
    printf("│ Dataset   │Build Ratio│Search Ratio│ Recall Loss │\n");
    printf("│ Type      │(x faster) │ (x faster) │ (%%)         │\n");
    printf("├───────────┼───────────┼────────────┼─────────────┤\n");
    printf("│ Random    │ %9.1fx │ %10.1fx │ %11.1f │\n",
           random_results.float32_build_time / random_results.quant_build_time,
           random_results.float32_search_time / random_results.quant_search_time,
           random_results.float32_recall - random_results.quantized_recall);
    printf("│ Clustered │ %9.1fx │ %10.1fx │ %11.1f │\n",
           clustered_results.float32_build_time / clustered_results.quant_build_time,
           clustered_results.float32_search_time / clustered_results.quant_search_time,
           clustered_results.float32_recall - clustered_results.quantized_recall);
    printf("└───────────┴───────────┴────────────┴─────────────┘\n");
}

int main() {
    printf("=== CAGRA QUANTIZATION COMPARISON ===\n\n");
    
    cagra_build_search_test("random", generate_random_dataset, &random_results);
    printf("\n");
    cagra_build_search_test("clustered", generate_clustered_dataset, &clustered_results);
    print_summary_table();
    
    return 0;
}
