package com.nvidia.cuvs;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.slf4j.Logger;

public class TestUtil {

  public static float[][] generateData(Random random, int rows, int cols) {
    float[][] data = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = random.nextFloat() * 100;
      }
    }
    return data;
  }

  public static List<List<Integer>> generateExpectedResults(int topK, float[][] dataset, float[][] queries,
      Logger log) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (float[] query : queries) {
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        for (int k = 0; k < dimensions; k++) {
          distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
        }
        distances.put(j, Math.sqrt(distance));
      }

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors = distances.entrySet().stream().sorted(Map.Entry.comparingByValue())
          .map(Map.Entry::getKey).toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 2, dataset.length)));
    }

    log.info("Expected results generated successfully.");
    return neighborsResult;
  }
}
