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
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.*;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.*;
import static java.lang.foreign.MemoryLayout.PathElement.*;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import java.io.OutputStream;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.TieredIndex;
import com.nvidia.cuvs.internal.DatasetImpl;
import com.nvidia.cuvs.TieredIndexParams;
import com.nvidia.cuvs.TieredIndexQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.common.SearchResultsImpl;
import com.nvidia.cuvs.internal.panama.cuvsTieredIndex;
import com.nvidia.cuvs.internal.panama.cuvsTieredIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraSearchParams;

public class TieredIndexImpl implements TieredIndex {
    private final IndexReference indexReference;
    private final CuVSResourcesImpl resources;
    private final TieredIndexParams params;
    private final TieredIndexType indexType;
    private final float[][] vectors;
    private final Dataset dataset;
    private float[][] extendVectors;
    private Dataset extendDataset;

    private boolean destroyed;

    // Method handles for native function calls
    private static final MethodHandle buildMethodHandle = downcallHandle(
            "build_tiered_index",
            FunctionDescriptor.of(
                    ADDRESS, // return type: int
                    ADDRESS, // dataset
                    C_LONG, // nRows
                    C_LONG, // nCols
                    ADDRESS, // resources
                    ADDRESS, // params
                    ADDRESS));

    private static final MethodHandle destroyMethodHandle = downcallHandle("destroy_tiered_index",
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

    private static final MethodHandle searchMethodHandle = downcallHandle("search_tiered_index",
            FunctionDescriptor.ofVoid(
                    ADDRESS, // resources
                    ADDRESS, // search_params
                    ADDRESS, // index
                    ADDRESS, // queries
                    C_INT, // topk
                    C_LONG, // n_queries
                    C_INT, // dimensions
                    ADDRESS, // neighbors
                    ADDRESS, // distances
                    ADDRESS // return_value
            ));

    private static final MethodHandle extendMethodHandle = downcallHandle("extend_tiered_index",
            FunctionDescriptor.ofVoid(
                    ADDRESS, // dataset
                    C_LONG, // nRows
                    C_LONG, // nCols
                    ADDRESS, // resources
                    ADDRESS, // index
                    ADDRESS // return_value
            ));

    private static class IndexReference {
        private final MemorySegment memorySegment;

        protected IndexReference(CuVSResourcesImpl resources) {
            memorySegment = cuvsTieredIndex.allocate(resources.getArena());
        }

        protected IndexReference(MemorySegment indexMemorySegment) {
            this.memorySegment = indexMemorySegment;
        }

        protected MemorySegment getMemorySegment() {
            return memorySegment;
        }
    }

    public TieredIndexImpl(TieredIndexParams params, float[][] vectors,
            Dataset dataset, CuVSResourcesImpl resources) throws Throwable {
        if (vectors == null && dataset == null)
            throw new IllegalArgumentException("Must provide vectors or dataset");
        if (vectors != null && vectors.length == 0)
            throw new IllegalArgumentException("Vectors array must not be empty");
        this.vectors = vectors;
        this.dataset = dataset;
        this.params = params;
        this.resources = resources;
        this.indexType = params.getCagraParams() != null ? TieredIndex.TieredIndexType.CAGRA : null;
        this.destroyed = false;
        this.indexReference = build();
    }

    private IndexReference build() throws Throwable {

        long nRows = dataset != null ? dataset.size() : vectors.length;
        long nCols = dataset != null ? dataset.dimensions() : vectors[0].length;

        MemorySegment indexParamsMemorySegment = params != null
                ? segmentFromIndexParams(resources, params)
                : MemorySegment.NULL;

        MemorySegment dataSeg = dataset != null ? ((DatasetImpl) dataset).seg
                : Util.buildMemorySegment(resources.getArena(), vectors);

        try (var arena = Arena.ofConfined()) {
            MemorySegment returnValue = arena.allocate(C_INT);
            // Call the native builder!
            var indexSeg = (MemorySegment) buildMethodHandle.invokeExact(
                    dataSeg,
                    nRows,
                    nCols,
                    resources.getMemorySegment(),
                    indexParamsMemorySegment,
                    returnValue);

            Util.checkError(returnValue.get(C_INT, 0L), "buildMethodHandle");

            return new IndexReference(indexSeg);
        }
    }

    private static MemorySegment segmentFromIndexParams(CuVSResourcesImpl resources, TieredIndexParams params) {
        MemorySegment seg = cuvsTieredIndexParams.allocate(resources.getArena());

        int metric = switch (params.getMetric()) {
            case L2 -> 0;
            case INNER_PRODUCT -> 1;
            default -> throw new IllegalArgumentException("Unsupported metric: " + params.getMetric());
        };
        cuvsTieredIndexParams.metric(seg, metric);

        cuvsTieredIndexParams.min_ann_rows(seg, params.getMinAnnRows());
        cuvsTieredIndexParams.create_ann_index_on_extend(seg, params.isCreateAnnIndexOnExtend());

        CagraIndexParams cagraParams = params.getCagraParams();
        if (cagraParams != null) {
            MemorySegment cagraParamsSeg = cuvsCagraIndexParams.allocate(resources.getArena());

            cuvsCagraIndexParams.intermediate_graph_degree(cagraParamsSeg, cagraParams.getIntermediateGraphDegree());
            cuvsCagraIndexParams.graph_degree(cagraParamsSeg, cagraParams.getGraphDegree());
            cuvsCagraIndexParams.build_algo(cagraParamsSeg, cagraParams.getCagraGraphBuildAlgo().ordinal());

            cuvsTieredIndexParams.cagra_params(seg, cagraParamsSeg);
        }

        return seg;
    }

    private MemorySegment segmentFromSearchParams(CagraSearchParams params) {
        MemorySegment seg = cuvsCagraSearchParams.allocate(resources.getArena());
        cuvsCagraSearchParams.max_queries(seg, params.getMaxQueries());
        cuvsCagraSearchParams.itopk_size(seg, params.getITopKSize());
        cuvsCagraSearchParams.max_iterations(seg, params.getMaxIterations());
        if (params.getCagraSearchAlgo() != null) {
            cuvsCagraSearchParams.algo(seg, params.getCagraSearchAlgo().value);
        }
        cuvsCagraSearchParams.team_size(seg, params.getTeamSize());
        cuvsCagraSearchParams.search_width(seg, params.getSearchWidth());
        cuvsCagraSearchParams.min_iterations(seg, params.getMinIterations());
        cuvsCagraSearchParams.thread_block_size(seg, params.getThreadBlockSize());
        if (params.getHashMapMode() != null) {
            cuvsCagraSearchParams.hashmap_mode(seg, params.getHashMapMode().value);
        }
        cuvsCagraSearchParams.hashmap_max_fill_rate(seg, params.getHashMapMaxFillRate());
        cuvsCagraSearchParams.num_random_samplings(seg, params.getNumRandomSamplings());
        cuvsCagraSearchParams.rand_xor_mask(seg, params.getRandXORMask());
        return seg;
    }

    @Override
    public void destroyIndex() throws Throwable {
        checkNotDestroyed();
        try (var arena = Arena.ofConfined()) {
            MemorySegment returnValue = arena.allocate(C_INT);
            destroyMethodHandle.invokeExact(indexReference.getMemorySegment(), returnValue);
            Util.checkError(returnValue.get(C_INT, 0L), "destroyIndexMethodHandle");
        } finally {
            destroyed = true;
        }
    }

    @Override
    public SearchResults search(TieredIndexQuery query) throws Throwable {
        checkNotDestroyed();
        long numQueries = query.getQueryVectors().length;
        int topK = query.getMapping() != null ? Math.min(query.getMapping().size(), query.getTopK()) : query.getTopK();
        int dimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;
        long numBlocks = (long) topK * numQueries;

        SequenceLayout neighborsLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
        SequenceLayout distancesLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
        MemorySegment neighborsSeg = resources.getArena().allocate(neighborsLayout);
        MemorySegment distancesSeg = resources.getArena().allocate(distancesLayout);
        MemorySegment queriesSeg = Util.buildMemorySegment(resources.getArena(), query.getQueryVectors());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment returnValue = arena.allocate(C_INT);
            searchMethodHandle.invokeExact(
                    resources.getMemorySegment(),
                    segmentFromSearchParams(query.getCagraSearchParameters()),
                    indexReference.getMemorySegment(),
                    queriesSeg,
                    topK,
                    numQueries,
                    dimension,
                    neighborsSeg,
                    distancesSeg,
                    returnValue);

            checkError(returnValue.get(C_INT, 0L), "searchMethodHandle");
        }
        return new TieredSearchResultsImpl(
                neighborsLayout, distancesLayout, neighborsSeg, distancesSeg, topK, query.getMapping(), numQueries);
    }

    private void checkNotDestroyed() {
        if (destroyed) {
            throw new IllegalStateException("Index already destroyed");
        }
    }

    @Override
    public ExtendBuilder extend() {
        checkNotDestroyed();
        return new ExtendBuilder(this);
    }

    private void performExtend() throws Throwable {
        long nRows = extendDataset != null ? extendDataset.size() : extendVectors.length;
        long nCols = extendDataset != null ? extendDataset.dimensions() : extendVectors[0].length;

        MemorySegment dataSeg = extendDataset != null ? ((DatasetImpl) extendDataset).seg
                : Util.buildMemorySegment(resources.getArena(), extendVectors);

        try (var arena = Arena.ofConfined()) {
            MemorySegment returnValue = arena.allocate(C_INT);

            // Fix parameter order to match C function
            extendMethodHandle.invokeExact(
                    dataSeg, // dataset
                    nRows, // rows
                    nCols, // dimensions
                    resources.getMemorySegment(), // resources
                    indexReference.getMemorySegment(), // index
                    returnValue // return_value
            );

            Util.checkError(returnValue.get(C_INT, 0L), "extendMethodHandle");
        }

        // Clear extend data after use
        extendVectors = null;
        extendDataset = null;
    }

    @Override
    public CuVSResources getCuVSResources() {
        return resources;
    }

    @Override
    public TieredIndexType getIndexType() {
        return indexType;
    }

    public TieredIndexParams getIndexParams() {
        return params;
    }

    @Override
    public TieredIndexParams getIndexParameters() {
        return getIndexParams();
    }

    public static TieredIndex.Builder newBuilder(CuVSResources cuvsResources) {
        Objects.requireNonNull(cuvsResources);
        if (!(cuvsResources instanceof CuVSResourcesImpl)) {
            throw new IllegalArgumentException("Unsupported " + cuvsResources);
        }
        return new Builder((CuVSResourcesImpl) cuvsResources);
    }

    public static class Builder implements TieredIndex.Builder {
        private final CuVSResourcesImpl resources;
        private float[][] vectors;
        private Dataset dataset;
        private TieredIndexParams params;
        private TieredIndexType indexType = TieredIndexType.CAGRA;

        private Builder(CuVSResourcesImpl resources) {
            this.resources = resources;
        }

        @Override
        public Builder from(InputStream inputStream) {
            throw new UnsupportedOperationException("Deserialization of TieredIndex is not yet supported");
        }

        @Override
        public Builder withDataset(float[][] vectors) {
            this.vectors = vectors;
            return this;
        }

        @Override
        public Builder withDataset(Dataset dataset) {
            this.dataset = dataset;
            return this;
        }

        @Override
        public Builder withIndexParams(TieredIndexParams params) {
            this.params = params;
            return this;
        }

        @Override
        public Builder withIndexType(TieredIndexType indexType) {
            this.indexType = indexType;
            return this;
        }

        @Override
        public TieredIndex build() throws Throwable {
            if (params == null) {
                throw new IllegalStateException("Index parameters must be provided");
            }
            return new TieredIndexImpl(params, vectors, dataset, resources);
        }
    }

    public static class ExtendBuilder implements TieredIndex.ExtendBuilder {
        private final TieredIndexImpl index;
        private float[][] vectors;
        private Dataset dataset;

        private ExtendBuilder(TieredIndexImpl index) {
            this.index = index;
        }

        @Override
        public ExtendBuilder withDataset(float[][] vectors) {
            this.vectors = vectors;
            return this;
        }

        @Override
        public ExtendBuilder withDataset(Dataset dataset) {
            this.dataset = dataset;
            return this;
        }

        @Override
        public void execute() throws Throwable {
            if (vectors != null && dataset != null) {
                throw new IllegalArgumentException(
                        "Please specify only one type of dataset (a float[][] or a Dataset instance)");
            }
            if (vectors == null && dataset == null) {
                throw new IllegalArgumentException("Must provide vectors or dataset");
            }

            index.extendVectors = vectors;
            index.extendDataset = dataset;
            index.performExtend();
        }
    }

}