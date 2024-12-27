/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuvs/neighbors/hnsw.h>

#define try bool __HadError=false;
#define catch(x) ExitJmp:if(__HadError)
#define throw(x) {__HadError=true;goto ExitJmp;}

cuvsResources_t create_resources(int *returnValue) {
  cuvsResources_t cuvsResources;
  *returnValue = cuvsResourcesCreate(&cuvsResources);
  return cuvsResources;
}

void destroy_resources(cuvsResources_t cuvsResources, int *returnValue) {
  *returnValue = cuvsResourcesDestroy(cuvsResources);
}

DLManagedTensor prepare_tensor(void *data, int64_t shape[], DLDataTypeCode code) {
  DLManagedTensor tensor;

  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = NULL;

  return tensor;
}

cuvsCagraIndex_t build_cagra_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvsResources, int *returnValue,
    cuvsCagraIndexParams_t index_params, cuvsCagraCompressionParams_t compression_params, int numWriterThreads) {

  omp_set_num_threads(numWriterThreads);
  cuvsRMMPoolMemoryResourceEnable(95, 95, true);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  index_params->compression = compression_params;
  *returnValue = cuvsCagraBuild(cuvsResources, index_params, &dataset_tensor, index);
    omp_set_num_threads(1);

  return index;
}

void destroy_cagra_index(cuvsCagraIndex_t index, int *returnValue) {
  *returnValue = cuvsCagraIndexDestroy(index);
}

void serialize_cagra_index(cuvsResources_t cuvsResources, cuvsCagraIndex_t index, int *returnValue, char* filename) {
  *returnValue = cuvsCagraSerialize(cuvsResources, filename, index, true);
}

void deserialize_cagra_index(cuvsResources_t cuvsResources, cuvsCagraIndex_t index, int *returnValue, char* filename) {
  *returnValue = cuvsCagraDeserialize(cuvsResources, filename, index);
}

void search_cagra_index(cuvsCagraIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvsResources, int *neighbors_h, float *distances_h, int *returnValue, cuvsCagraSearchParams_t search_params) {

  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(cuvsResources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvsResources, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(cuvsResources, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat);

  *returnValue = cuvsCagraSearch(cuvsResources, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvsResources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvsResources, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(cuvsResources, queries_d, sizeof(float) * n_queries * dimensions);
}

void convert_cagra_to_hnsw(cuvsResources_t resources,
                           const char* cagra_filename,
                           const char* hnsw_filename,
                           int* return_value) {
    if (!resources || !cagra_filename || !hnsw_filename) {
        *return_value = -1; // Invalid parameters
        printf("Error: Invalid parameters provided\n");
        return;
    }

    printf("Initializing conversion: CAGRA -> HNSW\n");
    printf("CAGRA file: %s\n", cagra_filename);
    printf("HNSW file: %s\n", hnsw_filename);

    cuvsCagraIndex_t cagra_index;
    cuvsCagraIndexCreate(&cagra_index);

    // Step 1: Deserialize the CAGRA index
    *return_value = cuvsCagraDeserialize(resources, cagra_filename, cagra_index);
    if (*return_value != CUVS_SUCCESS) {
        printf("Error: Failed to deserialize CAGRA index. Error code: %d\n", *return_value);
        cuvsCagraIndexDestroy(cagra_index);
        return;
    }
    printf("Successfully deserialized CAGRA index\n");

    // Step 2: Serialize to HNSW format
    *return_value = cuvsCagraSerializeToHnswlib(resources, hnsw_filename, cagra_index);
    if (*return_value != CUVS_SUCCESS) {
        printf("Error: Failed to serialize to HNSW format. Error code: %d\n", *return_value);
        cuvsCagraIndexDestroy(cagra_index);
        return;
    }   

    printf("Successfully serialized CAGRA index to HNSW format: %s\n", hnsw_filename);
    cuvsCagraIndexDestroy(cagra_index);
}

void search_hnsw_index(cuvsHnswIndex_t index, 
                       float* queries, 
                       int topk, 
                       long n_queries, 
                       int dimensions, 
                       cuvsResources_t cuvsResources, 
                       uint64_t* neighbors_h, 
                       float* distances_h, 
                       int* returnValue, 
                       cuvsHnswSearchParams_t search_params) {

    printf("C layer function invoked successfully\n");
    uint64_t* neighbors;
    float* distances;
    float* queries_d;

    // Allocate memory
    cuvsRMMAlloc(cuvsResources, (void**)&queries_d, sizeof(float) * n_queries * dimensions);
    cuvsRMMAlloc(cuvsResources, (void**)&neighbors, sizeof(uint64_t) * n_queries * topk);
    cuvsRMMAlloc(cuvsResources, (void**)&distances, sizeof(float) * n_queries * topk);

    // Copy queries to device memory
    cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

    // Prepare tensors
    int64_t queries_shape[2] = {n_queries, dimensions};
    DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat);

    int64_t neighbors_shape[2] = {n_queries, topk};
    DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt);

    int64_t distances_shape[2] = {n_queries, topk};
    DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat);

    // Perform search
    *returnValue = cuvsHnswSearch(cuvsResources, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);

    // Check and print results
    if (*returnValue != CUVS_SUCCESS) {
        printf("Error: Search failed. Error code: %d\n", *returnValue);
    } else {
        printf("Search completed successfully\n");

        // Print neighbors and distances
        printf("HNSW Search Results:\n");
        for (long i = 0; i < n_queries; i++) {
            printf("Query %ld:\n", i);
            printf("  Neighbors: ");
            for (int j = 0; j < topk; j++) {
                printf("%lu ", neighbors[i * topk + j]);
            }
            printf("\n");

            printf("  Distances: ");
            for (int j = 0; j < topk; j++) {
                printf("%.6f ", distances[i * topk + j]);
            }
            printf("\n");
        }
    }

    // Copy results back to host memory
    cudaMemcpy(neighbors_h, neighbors, sizeof(uint64_t) * n_queries * topk, cudaMemcpyDefault);
    cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

    // Free allocated resources
    cuvsRMMFree(cuvsResources, distances, sizeof(float) * n_queries * topk);
    cuvsRMMFree(cuvsResources, neighbors, sizeof(uint64_t) * n_queries * topk);
    cuvsRMMFree(cuvsResources, queries_d, sizeof(float) * n_queries * dimensions);
}


