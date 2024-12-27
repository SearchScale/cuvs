package com.nvidia.cuvs.hnsw;

import java.lang.foreign.MemorySegment;

public class HnswQuery {
  private final float[][] queryVectors;
  private final int topK;
  private final HnswSearchParameters searchParameters;
  private final MemorySegment indexMemorySegment;

  public HnswQuery(MemorySegment indexMemorySegment, float[][] queryVectors, int topK,
      HnswSearchParameters searchParams) {
    this.queryVectors = queryVectors;
    this.topK = topK;
    this.searchParameters = searchParams;
    this.indexMemorySegment = indexMemorySegment;
  }

  public float[][] getQueryVectors() {
    return queryVectors;
  }

  public int getTopK() {
    return topK;
  }

  public HnswSearchParameters getSearchParameters() {
    return searchParameters;
  }

  public MemorySegment getIndexMemorySegment() {
    return indexMemorySegment;
  }

}
