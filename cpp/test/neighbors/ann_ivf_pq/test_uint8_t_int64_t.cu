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

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_u08_i64 = ivf_pq_test<float, uint8_t, int64_t>;

TEST_BUILD_SEARCH(f32_u08_i64)
TEST_BUILD_EXTEND_SEARCH(f32_u08_i64)
INSTANTIATE(f32_u08_i64, small_dims_per_cluster() + enum_variety());

}  // namespace cuvs::neighbors::ivf_pq
