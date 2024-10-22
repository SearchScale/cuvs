#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

cuvsResources_t create_resource() {
  cuvsResources_t res;  
  int rx = cuvsResourcesCreate(&res);
  return res;
}
                       
cuvsCagraIndex_t build_index(float *dataset, long rows, long dimension, cuvsResources_t res) {
  
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim = 2;
  dataset_tensor.dl_tensor.dtype.code = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits = 32;
  dataset_tensor.dl_tensor.dtype.lanes = 1;
  int64_t dataset_shape[2] = {rows, dimension};
  dataset_tensor.dl_tensor.shape = dataset_shape;
  dataset_tensor.dl_tensor.strides = NULL;

  cuvsCagraIndexParams_t index_params;
  cuvsCagraIndexParamsCreate(&index_params);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  int resx = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
  
  return index;
}

void search_index(cuvsCagraIndex_t index, float *queries, int topk, long n_queries, long dimension, 
    cuvsResources_t res, int *neighbors_h, float *distances_h) {

  int64_t n_cols = dimension;
  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(res, (void**) &queries_d, sizeof(float) * n_queries * n_cols);
  cuvsRMMAlloc(res, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(res, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * 4 * 2, cudaMemcpyDefault);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data = queries_d;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim = 2;
  queries_tensor.dl_tensor.dtype.code = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits = 32;
  queries_tensor.dl_tensor.dtype.lanes = 1;
  int64_t queries_shape[2] = {n_queries, n_cols};
  queries_tensor.dl_tensor.shape = queries_shape;
  queries_tensor.dl_tensor.strides = NULL;

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data = neighbors;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim = 2;
  neighbors_tensor.dl_tensor.dtype.code = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits = 32;
  neighbors_tensor.dl_tensor.dtype.lanes = 1;
  int64_t neighbors_shape[2] = {n_queries, topk};
  neighbors_tensor.dl_tensor.shape = neighbors_shape;
  neighbors_tensor.dl_tensor.strides = NULL;

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data = distances;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim = 2;
  distances_tensor.dl_tensor.dtype.code = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits = 32;
  distances_tensor.dl_tensor.dtype.lanes = 1;
  int64_t distances_shape[2] = {n_queries, topk};
  distances_tensor.dl_tensor.shape = distances_shape;
  distances_tensor.dl_tensor.strides = NULL;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);

  int rs = cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);

}
