package com.nvidia.cuvs;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;

@RunWith(RandomizedRunner.class)
public class BruteforceRandomizedTest extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    initializeRandom();
    log.info("Random context initialized for test.");
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    for (int i = 0; i < 10; i++) {
      tmpResultsTopKWithRandomValues();
    }
  }

  public void tmpResultsTopKWithRandomValues() throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize = random.nextInt(DATASET_SIZE_LIMIT) + 1;
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);

    if (datasetSize < topK)
      datasetSize = topK;

    // Generate a random dataset
    float[][] dataset = TestUtil.generateData(random, datasetSize, dimensions);

    // Generate random query vectors
    float[][] queries = TestUtil.generateData(random, numQueries, dimensions);

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
    List<List<Integer>> expected = TestUtil.generateExpectedResults(topK, dataset, queries, log);

    // Create CuVS index and query
    try (CuVSResources resources = new CuVSResources()) {

      BruteForceQuery query = new BruteForceQuery.Builder()
          .withTopK(topK)
          .withQueryVectors(queries)
          .build();

      BruteForceIndexParams indexParams = new BruteForceIndexParams.Builder()
          .withNumWriterThreads(32)
          .build();

      BruteForceIndex index = new BruteForceIndex.Builder(resources)
          .withDataset(dataset)
          .withIndexParams(indexParams)
          .build();

      log.info("Index built successfully. Executing search...");
      BruteForceSearchResults results = index.search(query);

      for (int i = 0; i < numQueries; i++) {
        log.info("Results returned for query " + i + ": " + results.getResults().get(i).keySet());
        log.info("Expected results for query " + i + ": " + expected.get(i).subList(0, Math.min(topK, datasetSize)));
      }

      // actual vs. expected results
      for (int i = 0; i < results.getResults().size(); i++) {
        Map<Integer, Float> result = results.getResults().get(i);
        assertEquals("TopK mismatch for query.", Math.min(topK, datasetSize), result.size());

        // Sort result by values (distances) and extract keys
        List<Integer> sortedResultKeys = result.entrySet().stream().sorted(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey) // Extract sorted keys
            .toList();

        // just make sure that the first 5 results are in the expected list (which
        // comprises of 2*topK results)
        for (int j = 0; j < Math.min(5, sortedResultKeys.size()); j++) {
          assertTrue("Not found in expected list: " + sortedResultKeys.get(j),
              expected.get(i).contains(sortedResultKeys.get(j)));
        }
      }
    }
  }
}
