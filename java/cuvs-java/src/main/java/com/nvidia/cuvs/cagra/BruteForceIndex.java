package com.nvidia.cuvs.cagra;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class BruteForceIndex {
    private final float[][] dataset;

    public BruteForceIndex(float[][] dataset) {
        this.dataset = dataset;
    }

    /**
     * Perform a brute force search for the top-K nearest neighbors.
     *
     * @param query The query object containing query vectors.
     * @return SearchResult object containing the top-K results for multiple queries.
     */
    public SearchResult search(CuVSQuery query) {
        int topK = query.getTopK();
        float[][] queries = query.getQueries();
        Map<Integer, Map<Integer, Float>> results = new HashMap<>();

        for (int queryIndex = 0; queryIndex < queries.length; queryIndex++) {
            float[] currentQuery = queries[queryIndex];
            Map<Integer, Float> distances = new HashMap<>();

            // Calculate distances to all dataset vectors
            for (int datasetIndex = 0; datasetIndex < dataset.length; datasetIndex++) {
                float distance = calculateEuclideanDistance(currentQuery, dataset[datasetIndex]);
                distances.put(datasetIndex, distance);
            }

            // Sort and keep the top-K results
            Map<Integer, Float> topKResults = distances.entrySet()
                    .stream()
                    .sorted(Map.Entry.comparingByValue())
                    .limit(topK)
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

            results.put(queryIndex, topKResults);
        }

        return new SearchResult(results);
    }

    /**
     * Calculate Euclidean distance between two vectors.
     *
     * @param vector1 First vector.
     * @param vector2 Second vector.
     * @return Euclidean distance.
     */
    private float calculateEuclideanDistance(float[] vector1, float[] vector2) {
        float sum = 0.0f;
        for (int i = 0; i < vector1.length; i++) {
            sum += Math.pow(vector1[i] - vector2[i], 2);
        }
        return (float) Math.sqrt(sum);
    }
}
