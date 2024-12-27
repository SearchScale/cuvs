package com.nvidia.cuvs.hnsw;

public class HnswSearchResults {
  private final long[][] neighbors;
  private final float[][] distances;
  private final int topK;
  private final int numQueries;

  public HnswSearchResults(long[][] neighbors, float[][] distances, int topK, int numQueries) {
    this.neighbors = neighbors;
    this.distances = distances;
    this.topK = topK;
    this.numQueries = numQueries;
  }

  public long[][] getNeighbors() {
    return neighbors;
  }

  public float[][] getDistances() {
    return distances;
  }

  public int getTopK() {
    return topK;
  }

  public int getNumQueries() {
    return numQueries;
  }
}
