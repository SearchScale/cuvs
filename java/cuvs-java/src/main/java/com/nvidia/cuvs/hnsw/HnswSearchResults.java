package com.nvidia.cuvs.hnsw;

public class HnswSearchResults {
  private final int[][] neighbors;
  private final float[][] distances;
  private final int topK;
  private final int numQueries;

  public HnswSearchResults(int[][] neighbors, float[][] distances, int topK, int numQueries) {
    this.neighbors = neighbors;
    this.distances = distances;
    this.topK = topK;
    this.numQueries = numQueries;
  }

  // Getters for validation
  public int[][] getNeighbors() {
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
