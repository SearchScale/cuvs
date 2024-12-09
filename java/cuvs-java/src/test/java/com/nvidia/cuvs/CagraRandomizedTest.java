package com.nvidia.cuvs;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedContext;

public class CagraRandomizedTest extends LuceneTestCase {
  private Random random;
  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {

    this.random = random();
    log.info("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    // Generate a random dataset
    int numRows = random.nextInt(10) + 1;
    int numCols = random.nextInt(5) + 1;
    float[][] dataset = new float[numRows][numCols];
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }

    // Generate random query vectors
    int numQueries = random.nextInt(5) + 1;
    float[][] queries = new float[numQueries][numCols];
    for (int i = 0; i < numQueries; i++) {
      for (int j = 0; j < numCols; j++) {
        queries[i][j] = random.nextFloat() * 100;
      }
    }

    int topK = random.nextInt(numRows) + 1;

    log.info("Dataset size: {}x{}", numRows, numCols);
    log.info("Query size: {}x{}", numQueries, numCols);
    log.info("TopK: {}", topK);

    log.info("Dataset:");
    for (float[] row : dataset) {
      log.info(java.util.Arrays.toString(row));
    }

    log.info("Queries:");
    for (float[] query : queries) {  
      log.info(java.util.Arrays.toString(query));
    }

    CuVSResources resources = new CuVSResources();

    CagraIndexParams indexParams = new CagraIndexParams.Builder(resources).build();

    CagraIndex index = new CagraIndex.Builder(resources).withDataset(dataset).withIndexParams(indexParams).build();

    CagraQuery query = new CagraQuery.Builder().withQueryVectors(queries).withTopK(topK)
        .withSearchParams(new CagraSearchParams.Builder(resources).build()).build();

    CagraSearchResults results = index.search(query);

    results.getResults().forEach(result -> {
      log.info("Result size: {}", result.size());
      assertEquals("TopK mismatch for query.", Math.min(topK, numRows), result.size());
    });
  }

}
