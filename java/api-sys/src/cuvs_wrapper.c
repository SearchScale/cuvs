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
                       
cuvsCagraIndex_t build_index(float *dataset, long rows, long dimension, cuvsResources_t res, int *rv) {
  
  int64_t dataset_shape[2] = {rows, dimension};
  DLManagedTensor dataset_tensor = prepare_tensor(dataset, dataset_shape, kDLFloat);

  cuvsCagraIndexParams_t index_params;
  cuvsCagraIndexParamsCreate(&index_params);

  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  *rv = cuvsCagraBuild(res, index_params, &dataset_tensor, index);
  return index;
}

void search_index(cuvsCagraIndex_t index, float *queries, int topk, long n_queries, long dimension, 
    cuvsResources_t res, int *neighbors_h, float *distances_h, int *rv) {

  uint32_t *neighbors;
  float *distances, *queries_d;
  cuvsRMMAlloc(res, (void**) &queries_d, sizeof(float) * n_queries * dimension);
  cuvsRMMAlloc(res, (void**) &neighbors, sizeof(uint32_t) * n_queries * topk);
  cuvsRMMAlloc(res, (void**) &distances, sizeof(float) * n_queries * topk);

  cudaMemcpy(queries_d, queries, sizeof(float) * 4 * 2, cudaMemcpyDefault);

  int64_t queries_shape[2] = {n_queries, dimension};
  DLManagedTensor queries_tensor = prepare_tensor(queries_d, queries_shape, kDLFloat);

  int64_t neighbors_shape[2] = {n_queries, topk};
  DLManagedTensor neighbors_tensor = prepare_tensor(neighbors, neighbors_shape, kDLUInt);

  int64_t distances_shape[2] = {n_queries, topk};
  DLManagedTensor distances_tensor = prepare_tensor(distances, distances_shape, kDLFloat);

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);

  *rv = cuvsCagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor,
                  &distances_tensor);

  cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault);
  cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault);
}
