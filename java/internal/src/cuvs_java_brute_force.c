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
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


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


cuvsResources_t create_resource(int *returnValue) {
  cuvsResources_t cuvsResources;
  *returnValue = cuvsResourcesCreate(&cuvsResources);
  return cuvsResources;
}


void destroy_resource(cuvsResources_t cuvsResources, int *returnValue) {
  *returnValue = cuvsResourcesDestroy(cuvsResources);
}


void destroy_brute_force_index(cuvsBruteForceIndex_t index, int *returnValue) {
  *returnValue = cuvsBruteForceIndexDestroy(index);
}


cuvsBruteForceIndex_t build_brute_force_index(float *dataset, long rows, long dimensions, cuvsResources_t cuvsResources, int *returnValue) {

  int64_t dataset_shape[2] = {rows, dimensions};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat);

  cuvsBruteForceIndex_t index;
  cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);

  *returnValue = cuvsBruteForceBuild(cuvsResources, &dataset_tensor, L2Expanded, 0.f, index);
  printf("brute force build return value: %d\n", *returnValue);
  return index;
}


void search_brute_force_index(cuvsBruteForceIndex_t index, float *queries, int topk, long n_queries, int dimensions, 
    cuvsResources_t cuvsResources, int *neighbors_h, float *distances_h, int *returnValue) {
  DLManagedTensor dataset_tensor;
  DLManagedTensor queries_tensor;
  DLManagedTensor neighbors_tensor;
  DLManagedTensor distances_tensor;
  DLManagedTensor bitmap_tensor;

  cuvsFilter prefilter;

  *returnValue = cuvsBruteForceSearch(cuvsResources, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);
}
