package com.nvidia.cuvs.hnsw;

public class HnswQuery {
  private final float[][] queryVectors;
  private final int topK;
  private final HnswSearchParameters searchParameters;

  public HnswQuery(float[][] queryVectors, int topK, HnswSearchParameters searchParams) {
    this.queryVectors = queryVectors;
    this.topK = topK;
    this.searchParameters = searchParams;
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

}
