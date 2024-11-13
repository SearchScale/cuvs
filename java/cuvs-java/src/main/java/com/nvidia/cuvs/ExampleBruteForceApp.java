package com.nvidia.cuvs;

import com.nvidia.cuvs.cagra.BruteForceIndex;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.SearchResult;

import java.util.Map;

public class ExampleBruteForceApp {

	public static void main(String[] args) throws Throwable {
    // Define dataset and queries
    float[][] dataset = {
            {1.0f, 2.0f},
            {2.0f, 3.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f},
            {7.0f, 8.0f}
    };

    float[][] queries = {
            {2.0f, 3.0f},
            {6.0f, 7.0f}
    };

    // Create query object
    CuVSQuery query = new CuVSQuery.Builder()
            .withTopK(3)
            .withQueryVectors(queries)
            .build();

    // Perform brute force search
    BruteForceIndex bruteForceIndex = new BruteForceIndex(dataset);
    SearchResult searchResult = bruteForceIndex.search(query);

    // Display results
    System.out.println("Brute Force Search Results:");
    Map<Integer, Map<Integer, Float>> allResults = searchResult.getAllResults();
    allResults.forEach((queryIndex, result) -> {
        System.out.println("Query " + queryIndex + ":");
        result.forEach((datasetIndex, distance) -> {
            System.out.println("Dataset Index: " + datasetIndex + ", Distance: " + distance);
        });
    });
}
}
