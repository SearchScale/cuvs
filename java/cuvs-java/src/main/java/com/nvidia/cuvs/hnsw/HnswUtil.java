package com.nvidia.cuvs.hnsw;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.common.Util;

public class HnswUtil {

  public static void serializeCagraToHnsw(CuVSResources resources, String cagraFilePath, String hnswFilePath) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment cagraPathSegment = Util.toCString(arena, cagraFilePath);
      MemorySegment hnswPathSegment = Util.toCString(arena, hnswFilePath);

      int result = (int) resources.cagraToHnswHandle.invokeExact(resources.getMemorySegment(), cagraPathSegment,
          hnswPathSegment);

      if (result != 0) {
        throw new RuntimeException("Failed to serialize CAGRA index to HNSW file. Error code: " + result);
      }
    } catch (Throwable e) {
      throw new RuntimeException("Error during serialization: " + e.getMessage(), e);
    }
  }

  public static HnswSearchResults search(CuVSResources resources, HnswQuery query) {
    try {
      // Extract parameters from query
      float[][] queryVectors = query.getQueryVectors();
      int topK = query.getTopK();
      int dimensions = queryVectors[0].length;
      int numQueries = queryVectors.length;

      // Use the shared arena from resources
      Arena arena = resources.arena;

      // Prepare flat memory for queries
      MemorySegment queriesMemory = Util.buildMemorySegment(resources.linker, resources.arena, queryVectors);
      MemorySegment neighborsMemory = arena.allocate(ValueLayout.JAVA_LONG.byteSize() * topK * numQueries);
      MemorySegment distancesMemory = arena.allocate(ValueLayout.JAVA_FLOAT.byteSize() * topK * numQueries);

      // Configure search parameters
      MemorySegment searchParamsMemory = arena.allocate(ValueLayout.JAVA_INT.byteSize() * 2);
      searchParamsMemory.setAtIndex(ValueLayout.JAVA_INT, 0, query.getSearchParameters().getEf());
      searchParamsMemory.setAtIndex(ValueLayout.JAVA_INT, 1, query.getSearchParameters().getNumThreads());

      // Invoke the HNSW search function using the native handle
      int result = (int) resources.hnswSearchHandle.invokeExact(resources.getMemorySegment(), // cuVS resources
          query.getIndexMemorySegment(), // cuvsHnswIndex_t
          queriesMemory, // Queries
          topK, // topK
          numQueries, // n_queries
          dimensions, // dimensions
          neighborsMemory, // Neighbors output
          distancesMemory, // Distances output
          searchParamsMemory // Search parameters
      );

      if (result != 0) {
        throw new RuntimeException("HNSW search failed. Error code: " + result);
      }

      // Extract neighbors and distances from memory
      long[] flatNeighbors = extractLongArray(neighborsMemory, topK * numQueries);
      float[] flatDistances = extractFloatArray(distancesMemory, topK * numQueries);

      long[][] neighbors = reshape(flatNeighbors, numQueries, topK);
      float[][] distances = reshape(flatDistances, numQueries, topK);

      return new HnswSearchResults(neighbors, distances, topK, numQueries);
    } catch (Throwable e) {
      throw new RuntimeException("Error during HNSW search", e);
    }
  }

  private static long[] extractLongArray(MemorySegment memory, int size) {
    long[] result = new long[size];
    for (int i = 0; i < size; i++) {
      result[i] = memory.get(ValueLayout.JAVA_LONG, i * ValueLayout.JAVA_LONG.byteSize());
    }
    return result;
  }

  private static float[] extractFloatArray(MemorySegment memory, int size) {
    float[] result = new float[size];
    for (int i = 0; i < size; i++) {
      result[i] = memory.get(ValueLayout.JAVA_FLOAT, i * ValueLayout.JAVA_FLOAT.byteSize());
    }
    return result;
  }

  private static long[][] reshape(long[] flatArray, int rows, int cols) {
    long[][] reshaped = new long[rows][cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(flatArray, i * cols, reshaped[i], 0, cols);
    }
    return reshaped;
  }

  private static float[][] reshape(float[] flatArray, int rows, int cols) {
    float[][] reshaped = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(flatArray, i * cols, reshaped[i], 0, cols);
    }
    return reshaped;
  }

}