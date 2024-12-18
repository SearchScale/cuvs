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

package com.nvidia.cuvs;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

import com.nvidia.cuvs.common.Util;
import com.nvidia.cuvs.panama.cuvsCagraIndex;

/**
 * {@link BruteForceIndex} ...
 * 
 * @since 25.02
 */
public class BruteForceIndex {

  private final float[][] dataset;
  private final CuVSResources resources;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle destroyIndexMethodHandle;
  private IndexReference bruteForceIndexReference;

  /*
   * Constructor for building the index using specified dataset
   */
  private BruteForceIndex(float[][] dataset, CuVSResources resources) throws Throwable {
    this.dataset = dataset;
    this.resources = resources;

    initializeMethodHandles();
    this.bruteForceIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   */
  private BruteForceIndex(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.dataset = null;
    this.resources = resources;

    initializeMethodHandles();
  }

  /**
   * Initializes the {@link MethodHandles} for invoking native methods.
   * 
   * @throws IOException @{@link IOException} is unable to load the native library
   */
  private void initializeMethodHandles() throws IOException {
    indexMethodHandle = resources.linker
        .downcallHandle(resources.getSymbolLookup().find("build_brute_force_index").get(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                resources.linker.canonicalLayouts().get("long"), resources.linker.canonicalLayouts().get("long"),
                ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("search_brute_force_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            resources.linker.canonicalLayouts().get("int"), resources.linker.canonicalLayouts().get("long"),
            resources.linker.canonicalLayouts().get("int"), ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    destroyIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("destroy_brute_force_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
  }

  public void destroyIndex() throws Throwable {
    MemoryLayout returnValueMemoryLayout = resources.linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    destroyIndexMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(), returnValueMemorySegment);
  }

  /**
   * Invokes the native build_index function via the Panama API to build the
   * {@link BruteForceIndex}
   * 
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = rows > 0 ? dataset[0].length : 0;

    MemoryLayout layout = resources.linker.canonicalLayouts().get("int");
    MemorySegment segment = resources.arena.allocate(layout);

    IndexReference indexReference = new IndexReference((MemorySegment) indexMethodHandle.invokeExact(
        Util.buildMemorySegment(resources.linker, resources.arena, dataset), rows, cols, resources.getMemorySegment(),
        segment));

    return indexReference;
  }

  /**
   * Invokes the native search_index via the Panama API for searching a CAGRA
   * index.
   * 
   * @param cuvsQuery an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link CagraSearchResults} containing the results
   */
  public BruteForceSearchResults search(BruteForceQuery cuvsQuery) throws Throwable {
    long numQueries = cuvsQuery.getQueryVectors().length;
    long numBlocks = cuvsQuery.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? cuvsQuery.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks,
        resources.linker.canonicalLayouts().get("int"));
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks,
        resources.linker.canonicalLayouts().get("float"));
    MemorySegment neighborsMemorySegment = resources.arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = resources.linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);

    searchMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(),
        Util.buildMemorySegment(resources.linker, resources.arena, cuvsQuery.getQueryVectors()), cuvsQuery.getTopK(),
        numQueries, vectorDimension, resources.getMemorySegment(), neighborsMemorySegment, distancesMemorySegment,
        returnValueMemorySegment);

    return new BruteForceSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), numQueries);
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder {

    private float[][] dataset;
    private CuVSResources cuvsResources;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     * 
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     * 
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     * 
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     * 
     * @return an instance of CagraIndex
     */
    public BruteForceIndex build() throws Throwable {
      if (inputStream != null) {
        return new BruteForceIndex(inputStream, cuvsResources);
      } else {
        return new BruteForceIndex(dataset, cuvsResources);
      }
    }
  }

  /**
   * Holds the memory reference to an index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResources resources) {
      memorySegment = cuvsCagraIndex.allocate(resources.arena);
    }

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
     * 
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
    }

    /**
     * Gets the instance of index MemorySegment.
     * 
     * @return index MemorySegment
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}
