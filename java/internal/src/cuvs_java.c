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
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

DLManagedTensor prepare_tensor(void *data, int64_t shape[], DLDataTypeCode code, int bits) {
  DLManagedTensor tensor;

  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = bits;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = NULL;

  return tensor;
}

cuvsCagraIndex_t build_cagra_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvsResources, int *returnValue,
    cuvsCagraIndexParams_t index_params, cuvsCagraCompressionParams_t compression_params, int numWriterThreads) {

  omp_set_num_threads(numWriterThreads);
  cuvsRMMPoolMemoryResourceEnable(95, 95, false);

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat, 32);

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
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt, 32);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32);

  *returnValue = cuvsCagraSearch(cuvsResources, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvsResources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvsResources, neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMFree(cuvsResources, queries_d, sizeof(float) * n_queries * dimensions);
}

void destroy_brute_force_index(cuvsBruteForceIndex_t index, int *returnValue) {
  *returnValue = cuvsBruteForceIndexDestroy(index);
  printf("brute force destroy return value: %d\n", *returnValue);
}

cuvsBruteForceIndex_t build_brute_force_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvsResources,
  int *returnValue) {

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat, 32);

  cuvsBruteForceIndex_t index;
  cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);

  *returnValue = cuvsBruteForceBuild(cuvsResources, &dataset_tensor, L2Expanded, 0.f, index);
  printf("brute force build return value: %d\n", *returnValue);
  return index;
}

void search_brute_force_index(cuvsBruteForceIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvsResources, int *neighbors_h, float *distances_h, int *returnValue) {
  int64_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(cuvsResources, (void**) &queries_d, sizeof(float) * n_queries * dimensions);
  cuvsRMMAlloc(cuvsResources, (void**) &neighbors, sizeof(int64_t) * n_queries * topk);
  cuvsRMMAlloc(cuvsResources, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * dimensions, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimensions};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat, 32);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLInt, 64);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat, 32);

  DLManagedTensor bitmap; //= prepare_tensor(NULL, NULL, kDLUInt, 32);
  // bitmap.dl_tensor.ndim = 1;
  cuvsFilter prefilter = {(uintptr_t)&bitmap, NO_FILTER};

  *returnValue = cuvsBruteForceSearch(cuvsResources, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);
  printf("brute force search return value: %d\n", *returnValue);
  printf("%s\n", cuvsGetLastErrorText());

  cudaMemcpy(neighbors_h, neighbors, sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

  cuvsRMMFree(cuvsResources, neighbors, sizeof(int64_t) * n_queries * topk);
  cuvsRMMFree(cuvsResources, distances, sizeof(float) * n_queries * topk);
  cuvsRMMFree(cuvsResources, queries_d, sizeof(float) * n_queries * dimensions);
}