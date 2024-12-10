package com.nvidia.cuvs.common;

public class GpuDetail {
  private final String name;
  private final long totalMemory;
  private final long freeMemory;

  public GpuDetail(String name, long totalMemory, long freeMemory) {
    this.name = name;
    this.totalMemory = totalMemory;
    this.freeMemory = freeMemory;
  }

  public String getName() {
    return name;
  }

  public long getTotalMemory() {
    return totalMemory;
  }

  public long getFreeMemory() {
    return freeMemory;
  }

  @Override
  public String toString() {
    return "GpuDetail{" + "name='" + name + '\'' + ", totalMemory=" + totalMemory + ", freeMemory=" + freeMemory + '}';
  }
}
