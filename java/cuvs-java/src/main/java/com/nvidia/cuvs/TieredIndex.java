/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
package com.nvidia.cuvs;

import java.io.InputStream;
import java.util.Objects;

import com.nvidia.cuvs.spi.CuVSProvider;

public interface TieredIndex {

    /**
     * Destroys the underlying native TieredIndex object.
     */
    void destroyIndex() throws Throwable;

    /**
     * Searches the index with the specified query and search parameters.
     *
     * @param query An instance of {@link TieredIndexQuery} describing the queries
     *              and search parameters.
     * @return An instance of {@link SearchResults} with neighbors/distances.
     */
    SearchResults search(TieredIndexQuery query) throws Throwable;

    /**
     * Returns the algorithm type backing this TieredIndex (e.g., CAGRA, IVF_FLAT,
     * IVF_PQ).
     */
    TieredIndexType getIndexType();

    /**
     * Returns the configuration parameters of this TieredIndex.
     */
    TieredIndexParams getIndexParameters();

    /**
     * Returns the resources handle associated with this TieredIndex.
     */
    CuVSResources getCuVSResources();

    /**
     * Creates a new Builder with an instance of CuVSResources.
     */
    static Builder newBuilder(CuVSResources cuvsResources) {
        Objects.requireNonNull(cuvsResources);
        return CuVSProvider.provider().newTieredIndexBuilder(cuvsResources);
    }

    // --- Builder interface for TieredIndex ---
    interface Builder {

        /**
         * (Optional) Deserializes an index from an InputStream.
         */
        Builder from(InputStream inputStream);

        /**
         * Sets the dataset vectors for building the TieredIndex.
         */
        Builder withDataset(float[][] vectors);

        /**
         * Sets the dataset for building the TieredIndex.
         */
        Builder withDataset(Dataset dataset);

        /**
         * Registers TieredIndex parameters with this Builder.
         */
        Builder withIndexParams(TieredIndexParams params);

        /**
         * Sets the index type (CAGRA, IVF_FLAT, IVF_PQ, etc).
         */
        Builder withIndexType(TieredIndexType indexType);

        /**
         * Builds and returns an instance of TieredIndex.
         */
        TieredIndex build() throws Throwable;
    }

    // --- Enum for supported TieredIndex types ---
    enum TieredIndexType {
        CAGRA
    }

    /**
     * Returns an ExtendBuilder to add new data to the index
     */
    ExtendBuilder extend();

    interface ExtendBuilder {
        ExtendBuilder withDataset(float[][] vectors);

        ExtendBuilder withDataset(Dataset dataset);

        void execute() throws Throwable;
    }
}
