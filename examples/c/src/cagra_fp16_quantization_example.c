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
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

// Use uint16_t to represent fp16 data (half precision floats are 16-bit)
typedef uint16_t fp16_t;

// IEEE 754 half precision conversion functions
fp16_t float_to_fp16(float f) {
    union { float f; uint32_t i; } u = { f };
    uint32_t i = u.i;
    
    uint32_t sign = (i >> 31) & 0x1;
    uint32_t exp = (i >> 23) & 0xff;
    uint32_t mant = i & 0x7fffff;
    
    // Handle special cases
    if (exp == 0) return (fp16_t)(sign << 15);  // Zero or denormal -> zero
    if (exp == 0xff) return (fp16_t)((sign << 15) | 0x7c00);  // Inf/NaN -> inf
    
    // Convert exponent
    int exp16 = (int)exp - 127 + 15;  // Convert from float32 bias to fp16 bias
    
    if (exp16 <= 0) return (fp16_t)(sign << 15);  // Underflow -> zero
    if (exp16 >= 31) return (fp16_t)((sign << 15) | 0x7c00);  // Overflow -> inf
    
    // Convert mantissa (round to nearest)
    uint32_t mant16 = (mant + 0x1000) >> 13;  // Round and shift
    if (mant16 >= 0x400) {  // Overflow in mantissa
        exp16++;
        mant16 = 0;
        if (exp16 >= 31) return (fp16_t)((sign << 15) | 0x7c00);  // Overflow -> inf
    }
    
    return (fp16_t)((sign << 15) | (exp16 << 10) | mant16);
}

float fp16_to_float(fp16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    
    union { float f; uint32_t i; } u;
    
    if (exp == 0) {
        if (mant == 0) {
            u.i = sign << 31;  // Zero
        } else {
            // Denormal -> normal
            exp = 127 - 15 + 1;  // Adjust bias and add 1
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            u.i = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u.i = (sign << 31) | (0xff << 23) | (mant << 13);  // Inf/NaN
    } else {
        u.i = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);  // Normal
    }
    
    return u.f;
}

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

// FP16 versions of the data
fp16_t dataset_fp16[N_ROWS][N_COLS];
fp16_t queries_fp16[N_QUERIES][N_COLS];

typedef struct {
    double float32_build_time, float32_search_time;
    double fp16_build_time, fp16_search_time;
    float float32_recall, fp16_recall;
    long float32_index_size, fp16_index_size;
} BenchmarkResults;

BenchmarkResults random_results, clustered_results;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

long get_file_size(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return -1;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
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

void convert_to_fp16() {
    printf("Converting float32 data to FP16...\n");
    
    // Convert dataset
    for (int i = 0; i < N_ROWS; ++i) {
        for (int j = 0; j < N_COLS; ++j) {
            dataset_fp16[i][j] = float_to_fp16(dataset[i][j]);
        }
    }
    
    // Convert queries
    for (int i = 0; i < N_QUERIES; ++i) {
        for (int j = 0; j < N_COLS; ++j) {
            queries_fp16[i][j] = float_to_fp16(queries[i][j]);
        }
    }
    
    printf("FP16 conversion completed.\n");
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

void evaluate_and_print_recall(uint32_t *float32_neighbors, uint32_t *fp16_neighbors, 
                              uint32_t **ground_truth, BenchmarkResults* results) {
    printf("\n=== RECALL EVALUATION ===\n");
    float total_float32_recall = 0.0f, total_fp16_recall = 0.0f;
    
    for (int q = 0; q < N_QUERIES; ++q) {
        float float32_recall = compute_recall(&float32_neighbors[q * TOPK], ground_truth[q]);
        float fp16_recall = compute_recall(&fp16_neighbors[q * TOPK], ground_truth[q]);
        
        printf("Query %d - Float32: %.1f%%, FP16: %.1f%%\n", q, float32_recall, fp16_recall);
        
        total_float32_recall += float32_recall;
        total_fp16_recall += fp16_recall;
    }
    
    float avg_float32_recall = total_float32_recall / N_QUERIES;
    float avg_fp16_recall = total_fp16_recall / N_QUERIES;
    results->float32_recall = avg_float32_recall; results->fp16_recall = avg_fp16_recall;
    
    printf("\n=== SUMMARY ===\n");
    printf("Float32 CAGRA average recall@%d: %.1f%%\n", TOPK, avg_float32_recall);
    printf("FP16 CAGRA average recall@%d: %.1f%%\n", TOPK, avg_fp16_recall);
    printf("Recall degradation due to FP16 quantization: %.1f%%\n", avg_float32_recall - avg_fp16_recall);
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
    search_params->itopk_size = 2 * TOPK;
    search_params->search_width = 20;
    search_params->max_iterations = 200;
    search_params->num_random_samplings = 16;
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
    float *distances, *queries_d;
    
    cuvsRMMAlloc(res, (void **)&queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsRMMAlloc(res, (void **)&neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMAlloc(res, (void **)&distances, sizeof(float) * N_QUERIES * TOPK);
    
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

    // Serialize the index to measure file size
    const char* float32_index_file = "/tmp/cagra_float32_index.bin";
    cuvsCagraSerialize(res, float32_index_file, index, true);
    results->float32_index_size = get_file_size(float32_index_file);
    printf("Float32 index serialized to %s (size: %ld bytes, %.2f MB)\n", 
           float32_index_file, results->float32_index_size, results->float32_index_size / (1024.0 * 1024.0));

    cuvsCagraSearchParamsDestroy(search_params);
    cuvsRMMFree(res, distances, sizeof(float) * N_QUERIES * TOPK);
    cuvsRMMFree(res, neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMFree(res, queries_d, sizeof(float) * N_QUERIES * N_COLS);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraIndexDestroy(index);

    return neighbors_h;
}

uint32_t* run_fp16_cagra_workflow(cuvsResources_t res, BenchmarkResults* results) {
    // Convert data to fp16 first
    convert_to_fp16();
    
    // Upload fp16 dataset to GPU
    fp16_t *dataset_fp16_d;
    cuvsRMMAlloc(res, (void **)&dataset_fp16_d, sizeof(fp16_t) * N_ROWS * N_COLS);
    cudaMemcpy(dataset_fp16_d, dataset_fp16, sizeof(fp16_t) * N_ROWS * N_COLS, cudaMemcpyHostToDevice);
    
    // Create tensor with actual fp16 data and specify it as 16-bit float
    DLManagedTensor fp16_dataset_tensor;
    int64_t dataset_shape[2] = {N_ROWS, N_COLS};
    prepare_tensor(&fp16_dataset_tensor, dataset_fp16_d, kDLCUDA, kDLFloat, 16, dataset_shape, 2);

    cuvsCagraIndexParams_t index_params = create_index_params();
    cuvsCagraIndex_t index;
    cuvsCagraIndexCreate(&index);

    double start_time = get_time();
    cuvsCagraBuild(res, index_params, &fp16_dataset_tensor, index);
    results->fp16_build_time = get_time() - start_time;
    printf("FP16 index build time: %.3f seconds\n", results->fp16_build_time);

    // Upload fp16 queries to GPU  
    fp16_t *queries_fp16_d;
    uint32_t *neighbors;
    float *distances;
    
    cuvsRMMAlloc(res, (void **)&queries_fp16_d, sizeof(fp16_t) * N_QUERIES * N_COLS);
    cuvsRMMAlloc(res, (void **)&neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMAlloc(res, (void **)&distances, sizeof(float) * N_QUERIES * TOPK);
    
    cudaMemcpy(queries_fp16_d, queries_fp16, sizeof(fp16_t) * N_QUERIES * N_COLS, cudaMemcpyHostToDevice);

    // Create tensors with actual fp16 data
    DLManagedTensor fp16_queries_tensor, neighbors_tensor, distances_tensor;
    int64_t queries_shape[2] = {N_QUERIES, N_COLS};
    int64_t neighbors_shape[2] = {N_QUERIES, TOPK};
    int64_t distances_shape[2] = {N_QUERIES, TOPK};
    
    prepare_tensor(&fp16_queries_tensor, queries_fp16_d, kDLCUDA, kDLFloat, 16, queries_shape, 2);
    prepare_tensor(&neighbors_tensor, neighbors, kDLCUDA, kDLUInt, 32, neighbors_shape, 2);
    prepare_tensor(&distances_tensor, distances, kDLCUDA, kDLFloat, 32, distances_shape, 2);

    cuvsCagraSearchParams_t search_params = create_search_params();
    cuvsFilter filter = {NO_FILTER, (uintptr_t)NULL};
    
    start_time = get_time();
    cuvsCagraSearch(res, search_params, index, &fp16_queries_tensor, &neighbors_tensor, &distances_tensor, filter);
    results->fp16_search_time = get_time() - start_time;
    printf("FP16 search time: %.3f seconds\n", results->fp16_search_time);

    uint32_t *neighbors_h = (uint32_t *)malloc(sizeof(uint32_t) * N_QUERIES * TOPK);
    cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * N_QUERIES * TOPK, cudaMemcpyDefault);

    // Serialize the index to measure file size
    const char* fp16_index_file = "/tmp/cagra_fp16_index.bin";
    cuvsCagraSerialize(res, fp16_index_file, index, true);
    results->fp16_index_size = get_file_size(fp16_index_file);
    printf("FP16 index serialized to %s (size: %ld bytes, %.2f MB)\n", 
           fp16_index_file, results->fp16_index_size, results->fp16_index_size / (1024.0 * 1024.0));

    cuvsCagraSearchParamsDestroy(search_params);
    cuvsRMMFree(res, distances, sizeof(float) * N_QUERIES * TOPK);
    cuvsRMMFree(res, neighbors, sizeof(uint32_t) * N_QUERIES * TOPK);
    cuvsRMMFree(res, queries_fp16_d, sizeof(fp16_t) * N_QUERIES * N_COLS);
    cuvsRMMFree(res, dataset_fp16_d, sizeof(fp16_t) * N_ROWS * N_COLS);
    cuvsCagraIndexParamsDestroy(index_params);
    cuvsCagraIndexDestroy(index);

    return neighbors_h;
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
    
    printf("\n--- FP16 Workflow ---\n");
    uint32_t *fp16_neighbors_h = run_fp16_cagra_workflow(res, results);
    
    evaluate_and_print_recall(neighbors_h, fp16_neighbors_h, ground_truth, results);

    free(fp16_neighbors_h);
    free(neighbors_h);
    for (int q = 0; q < N_QUERIES; ++q) {
        free(ground_truth[q]);
    }
    free(ground_truth);
    cuvsResourcesDestroy(res);
}

void print_summary_table() {
    printf("\n\n=== PERFORMANCE SUMMARY TABLE ===\n");
    printf("┌───────────┬─────┬─────┬─────┬───────────┬───────────┬───────────┬───────────┬─────────┬─────────┐\n");
    printf("│ Dataset   │  N  │ Dim │TopK │ F32 Build │ F32 Search│F16 Build  │F16 Search │ F32 Rec │F16 Rec  │\n");
    printf("│ Type      │ (k) │     │     │ (sec)     │ (sec)     │ (sec)     │ (sec)     │ (%%)     │ (%%)     │\n");
    printf("├───────────┼─────┼─────┼─────┼───────────┼───────────┼───────────┼───────────┼─────────┼─────────┤\n");
    printf("│ Random    │ %3d │%4d │ %3d │ %9.3f │ %9.3f │ %9.3f │ %9.3f │ %7.1f │ %7.1f │\n",
           N_ROWS/1000, N_COLS, TOPK,
           random_results.float32_build_time, random_results.float32_search_time,
           random_results.fp16_build_time, random_results.fp16_search_time,
           random_results.float32_recall, random_results.fp16_recall);
    printf("│ Clustered │ %3d │%4d │ %3d │ %9.3f │ %9.3f │ %9.3f │ %9.3f │ %7.1f │ %7.1f │\n",
           N_ROWS/1000, N_COLS, TOPK,
           clustered_results.float32_build_time, clustered_results.float32_search_time,
           clustered_results.fp16_build_time, clustered_results.fp16_search_time,
           clustered_results.float32_recall, clustered_results.fp16_recall);
    printf("└───────────┴─────┴─────┴─────┴───────────┴───────────┴───────────┴───────────┴─────────┴─────────┘\n");
    
    printf("\n=== SPEEDUP ANALYSIS ===\n");
    printf("┌───────────┬───────────┬────────────┬─────────────┐\n");
    printf("│ Dataset   │Build Ratio│Search Ratio│ Recall Loss │\n");
    printf("│ Type      │(x faster) │ (x faster) │ (%%)         │\n");
    printf("├───────────┼───────────┼────────────┼─────────────┤\n");
    printf("│ Random    │ %9.1fx │ %10.1fx │ %11.1f │\n",
           random_results.float32_build_time / random_results.fp16_build_time,
           random_results.float32_search_time / random_results.fp16_search_time,
           random_results.float32_recall - random_results.fp16_recall);
    printf("│ Clustered │ %9.1fx │ %10.1fx │ %11.1f │\n",
           clustered_results.float32_build_time / clustered_results.fp16_build_time,
           clustered_results.float32_search_time / clustered_results.fp16_search_time,
           clustered_results.float32_recall - clustered_results.fp16_recall);
    printf("└───────────┴───────────┴────────────┴─────────────┘\n");
    
    printf("\n=== INDEX SIZE COMPARISON ===\n");
    printf("┌───────────┬─────────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Dataset   │F32 Size (MB)│F16 Size (MB)│Size Ratio   │ Savings (%%) │\n");
    printf("│ Type      │             │             │(F16/F32)    │             │\n");
    printf("├───────────┼─────────────┼─────────────┼─────────────┼─────────────┤\n");
    printf("│ Random    │ %11.2f │ %11.2f │ %11.3f │ %11.1f │\n",
           random_results.float32_index_size / (1024.0 * 1024.0),
           random_results.fp16_index_size / (1024.0 * 1024.0),
           (double)random_results.fp16_index_size / random_results.float32_index_size,
           (1.0 - (double)random_results.fp16_index_size / random_results.float32_index_size) * 100.0);
    printf("│ Clustered │ %11.2f │ %11.2f │ %11.3f │ %11.1f │\n",
           clustered_results.float32_index_size / (1024.0 * 1024.0),
           clustered_results.fp16_index_size / (1024.0 * 1024.0),
           (double)clustered_results.fp16_index_size / clustered_results.float32_index_size,
           (1.0 - (double)clustered_results.fp16_index_size / clustered_results.float32_index_size) * 100.0);
    printf("└───────────┴─────────────┴─────────────┴─────────────┴─────────────┘\n");
}

int main() {
    printf("=== CAGRA FP16 QUANTIZATION COMPARISON ===\n\n");
    printf("This example demonstrates FP16 quantization by converting float32 data to\n");
    printf("actual 16-bit half-precision format (__half) and building CAGRA index with FP16 data.\n");
    printf("It compares Float32 CAGRA vs. genuine FP16 CAGRA performance and accuracy.\n\n");
    
    cagra_build_search_test("random", generate_random_dataset, &random_results);
    printf("\n");
    cagra_build_search_test("clustered", generate_clustered_dataset, &clustered_results);
    print_summary_table();
    
    return 0;
}