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
      // Extract query parameters
      float[][] queryVectors = query.getQueryVectors();
      int topK = query.getTopK();
      HnswSearchParameters searchParams = query.getSearchParameters();

      int numQueries = queryVectors.length;

      // Prepare tensors and allocate memory for results
      MemorySegment queryTensor = toTensor(resources.arena, queryVectors);
      MemorySegment neighborsMemory = resources.arena.allocate(ValueLayout.JAVA_INT.byteSize() * topK * numQueries);
      MemorySegment distancesMemory = resources.arena.allocate(ValueLayout.JAVA_FLOAT.byteSize() * topK * numQueries);

      // Configure search parameters
      MemorySegment efSegment = resources.arena.allocate(ValueLayout.JAVA_INT.byteSize());
      efSegment.set(ValueLayout.JAVA_INT, 0, searchParams.getEf());

      MemorySegment numThreadsSegment = resources.arena.allocate(ValueLayout.JAVA_INT.byteSize());
      numThreadsSegment.set(ValueLayout.JAVA_INT, 0, searchParams.getNumThreads());

      // Perform the search using the native handle
      int result = (int) resources.hnswSearchHandle.invokeExact(resources.getMemorySegment(), // Resources
          queryTensor, // Query tensor
          neighborsMemory, // Neighbors output
          distancesMemory, // Distances output
          efSegment, // Search parameter (ef)
          numThreadsSegment // Search parameter (numThreads)
      );

      if (result != 0) {
        throw new RuntimeException("Failed to perform HNSW search. Error code: " + result);
      }

      // Extract results from memory segments
      int[] flatNeighbors = new int[topK * numQueries];
      float[] flatDistances = new float[topK * numQueries];

      for (int i = 0; i < flatNeighbors.length; i++) {
        flatNeighbors[i] = neighborsMemory.getAtIndex(ValueLayout.JAVA_INT, i);
      }
      for (int i = 0; i < flatDistances.length; i++) {
        flatDistances[i] = distancesMemory.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      }

      // Reshape the flat results into 2D arrays
      int[][] neighbors = reshape(flatNeighbors, numQueries, topK);
      float[][] distances = reshape(flatDistances, numQueries, topK);

      return new HnswSearchResults(neighbors, distances, topK, numQueries);
    } catch (Throwable e) {
      throw new RuntimeException("Error during HNSW search: " + e.getMessage(), e);
    }
  }

  private static MemorySegment toTensor(Arena arena, float[][] vectors) {
    int rows = vectors.length;
    int cols = vectors[0].length;
    MemorySegment tensor = arena.allocate(ValueLayout.JAVA_FLOAT.byteSize() * rows * cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        tensor.setAtIndex(ValueLayout.JAVA_FLOAT, i * cols + j, vectors[i][j]);
      }
    }

    return tensor;
  }

  private static int[][] reshape(int[] flatArray, int rows, int cols) {
    int[][] reshaped = new int[rows][cols];
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