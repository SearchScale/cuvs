package com.nvidia.cuvs;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;

@RunWith(RandomizedRunner.class)
public class CagraRandomizedTest extends CuVSTestCase {

    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @Before
    public void setup() {
        initializeRandom();
        log.info("Random context initialized for test.");
    }

    @Test
    public void testResultsTopKWithRandomValues() throws Throwable {
        // Use old-style random generation logic
        int datasetSize = random.nextInt(400) + 1;
        int dimensions = random.nextInt(500) + 1;
        int numQueries = random.nextInt(500) + 1;
        int topK = random.nextInt(datasetSize) + 1;

        // Generate a random dataset
        float[][] dataset = new float[datasetSize][dimensions];
        for (int i = 0; i < datasetSize; i++) {
            for (int j = 0; j < dimensions; j++) {
                dataset[i][j] = random.nextFloat() * 100;
            }
        }

        // Generate random query vectors
        float[][] queries = new float[numQueries][dimensions];
        for (int i = 0; i < numQueries; i++) {
            for (int j = 0; j < dimensions; j++) {
                queries[i][j] = random.nextFloat() * 100;
            }
        }

        log.info("Dataset size: {}x{}", datasetSize, dimensions);
        log.info("Query size: {}x{}", numQueries, dimensions);
        log.info("TopK: {}", topK);

        // Debugging: Log dataset and queries
        if (log.isDebugEnabled()) {
            log.debug("Dataset:");
            for (float[] row : dataset) {
                log.debug(java.util.Arrays.toString(row));
            }

            log.debug("Queries:");
            for (float[] query : queries) {
                log.debug(java.util.Arrays.toString(query));
            }
        }

        // Sanity checks
        assert dataset.length > 0 : "Dataset is empty.";
        assert queries.length > 0 : "Queries are empty.";
        assert dimensions > 0 : "Invalid dimensions.";
        assert topK > 0 && topK <= datasetSize : "Invalid topK value.";

        // Generate expected results using brute force
        List<List<Integer>> expected = generateExpectedResults(topK, dataset, queries);

        // Create CuVS index and query
        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources).build();
        CagraIndex index = new CagraIndex.Builder(resources).withDataset(dataset).withIndexParams(indexParams).build();

        log.info("Index built successfully.");

        CagraQuery query = new CagraQuery.Builder()
                .withQueryVectors(queries)
                .withTopK(topK)
                .withSearchParams(new CagraSearchParams.Builder(resources).build())
                .build();

        log.info("Query built successfully. Executing search...");

        // Execute search and retrieve results
        CagraSearchResults results = index.search(query);

        // actual vs. expected results
        for (int i = 0; i < results.getResults().size(); i++) {
          Map<Integer, Float> result = results.getResults().get(i);
          log.info("Actual result for query {}: {}", i, result.keySet());
          log.info("Expected result for query {}: {}", i, expected.get(i));

          assertEquals("TopK mismatch for query.", Math.min(topK, datasetSize), result.size());

          // Sort result by values (distances) and extract keys
          List<Integer> sortedResultKeys = result.entrySet().stream()
              .sorted(Map.Entry.comparingByValue()) // Sort by value (distance)
              .map(Map.Entry::getKey) // Extract sorted keys
              .toList();

          log.info("Sorted Actual result for query {}: {}", i, sortedResultKeys);

          // Compare using primitive int arrays
          assertArrayEquals(
              "Query " + i + " mismatched",
              expected.get(i).stream().mapToInt(Integer::intValue).toArray(),
              sortedResultKeys.stream().mapToInt(Integer::intValue).toArray()
          );
      }

    }

    private List<List<Integer>> generateExpectedResults(int topK, float[][] dataset, float[][] queries) {
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
            List<Integer> neighbors = distances.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .toList();
            neighborsResult.add(neighbors.subList(0, Math.min(topK, dataset.length)));
        }

        log.info("Expected results generated successfully.");
        return neighborsResult;
    }
}
