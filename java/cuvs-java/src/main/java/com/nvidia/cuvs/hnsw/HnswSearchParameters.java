package com.nvidia.cuvs.hnsw;

public class HnswSearchParameters {

  private int ef;
  private int numThreads;

  public HnswSearchParameters(int ef, int numThreads) {
    this.ef = ef;
    this.numThreads = numThreads;
  }

  public HnswSearchParameters withEf(int ef) {
    this.ef = ef;
    return this;
  }

  public HnswSearchParameters withNumThreads(int threads) {
    this.numThreads = threads;
    return this;
  }

  public int getEf() {
    return ef;
  }

  public int getNumThreads() {
    return numThreads;
  }
}