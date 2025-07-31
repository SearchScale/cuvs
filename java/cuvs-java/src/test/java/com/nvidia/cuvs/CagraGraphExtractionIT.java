/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integration tests for CAGRA graph extraction and recreation functionality.
 *
 * This test suite verifies that:
 * 1. getGraph() method correctly extracts the graph from a CAGRA index
 * 2. from(graph, dataset, metric) method can recreate an index from extracted components
 * 3. Original and recreated indexes produce identical search results
 */
@RunWith(RandomizedRunner.class)
public class CagraGraphExtractionIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Random context initialized for graph extraction test.");
  }

  /**
   * Test basic graph extraction and recreation with small dataset.
   */
  @Test
  public void testBasicGraphExtractionAndRecreation() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Build original index
      CagraIndex originalIndex = buildCagraIndex(dataset, resources);

      // Extract graph from original index
      log.info("Extracting graph from original index...");
      int[][] extractedGraph = originalIndex.getGraph();
      assertNotNull("Extracted graph should not be null", extractedGraph);
      assertEquals(
          "Graph should have same number of rows as dataset",
          dataset.length,
          extractedGraph.length);
      assertTrue("Graph should have positive degree", extractedGraph[0].length > 0);

      log.info("Extracted graph shape: {}x{}", extractedGraph.length, extractedGraph[0].length);

      // Recreate index from extracted graph and dataset
      log.info("Recreating index from extracted graph...");
      CagraIndex recreatedIndex =
          CagraIndex.newBuilder(resources)
              .from(extractedGraph, dataset, CuvsDistanceType.L2Expanded)
              .build();

      // Compare search results
      compareSearchResults(originalIndex, recreatedIndex, queries, resources);

      // Cleanup
      originalIndex.destroyIndex();
      recreatedIndex.destroyIndex();
    }
  }

  /**
   * Test graph extraction and recreation with larger dataset.
   */
  @Test
  public void testGraphExtractionWithLargerDataset() throws Throwable {
    // Generate larger dataset for more comprehensive testing
    int numVectors = 10000;
    int dimensions = 128;
    float[][] dataset = generateData(random, numVectors, dimensions);
    float[][] queries = generateData(random, 10, dimensions);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Build original index with more complex parameters
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withNumWriterThreads(8)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex originalIndex =
          CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();

      // Extract and validate graph
      log.info("Extracting graph from larger dataset index...");
      int[][] extractedGraph = originalIndex.getGraph();
      assertNotNull("Extracted graph should not be null", extractedGraph);
      assertEquals("Graph should have correct number of rows", numVectors, extractedGraph.length);
      assertEquals("Graph should have correct degree", 64, extractedGraph[0].length);

      // Validate graph structure
      validateGraphStructure(extractedGraph, numVectors);

      // Recreate index
      log.info("Recreating index from extracted graph...");
      CagraIndex recreatedIndex =
          CagraIndex.newBuilder(resources)
              .from(extractedGraph, dataset, CuvsDistanceType.L2Expanded)
              .build();

      // Compare search results
      compareSearchResults(originalIndex, recreatedIndex, queries, resources);

      // Cleanup
      originalIndex.destroyIndex();
      recreatedIndex.destroyIndex();
    }
  }

  /**
   * Test that graph extraction fails gracefully for destroyed index.
   */
  @Test
  public void testGraphExtractionFromDestroyedIndex() throws Throwable {
    float[][] dataset = createSampleData();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndex index = buildCagraIndex(dataset, resources);

      // Destroy the index
      index.destroyIndex();

      // Attempt to extract graph from destroyed index should fail
      try {
        index.getGraph();
        fail("Should have thrown exception when extracting graph from destroyed index");
      } catch (IllegalStateException e) {
        assertTrue(
            "Exception message should indicate destroyed state",
            e.getMessage().contains("destroyed"));
      }
    }
  }

  /**
   * Test multiple extract-recreate cycles.
   */
  @Test
  public void testMultipleExtractionCycles() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndex currentIndex = buildCagraIndex(dataset, resources);

      // Perform multiple extract-recreate cycles
      for (int cycle = 0; cycle < 3; cycle++) {
        log.info("Extraction cycle {}", cycle + 1);

        // Extract graph
        int[][] extractedGraph = currentIndex.getGraph();

        // Recreate index
        CagraIndex nextIndex =
            CagraIndex.newBuilder(resources)
                .from(extractedGraph, dataset, CuvsDistanceType.L2Expanded)
                .build();

        // Compare with original
        compareSearchResults(currentIndex, nextIndex, queries, resources);

        // Cleanup previous index and use the new one for next iteration
        currentIndex.destroyIndex();
        currentIndex = nextIndex;
      }

      // Final cleanup
      currentIndex.destroyIndex();
    }
  }

  // Helper methods

  private CagraIndex buildCagraIndex(float[][] dataset, CuVSResources resources) throws Throwable {
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(2)
            .withIntermediateGraphDegree(4)
            .withNumWriterThreads(2)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    return CagraIndex.newBuilder(resources)
        .withDataset(dataset)
        .withIndexParams(indexParams)
        .build();
  }

  private void compareSearchResults(
      CagraIndex index1, CagraIndex index2, float[][] queries, CuVSResources resources)
      throws Throwable {
    CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

    CagraQuery query =
        new CagraQuery.Builder()
            .withTopK(3)
            .withSearchParams(searchParams)
            .withQueryVectors(queries)
            .withMapping(SearchResults.IDENTITY_MAPPING)
            .build();

    SearchResults results1 = index1.search(query);
    SearchResults results2 = index2.search(query);

    log.info("Original index results: {}", results1.getResults());
    log.info("Recreated index results: {}", results2.getResults());

    assertEquals(
        "Search results should be identical", results1.getResults(), results2.getResults());
  }

  private void validateGraphStructure(int[][] graph, int numVectors) {
    // Validate that all graph entries are valid vector indices
    for (int i = 0; i < graph.length; i++) {
      for (int j = 0; j < graph[i].length; j++) {
        int neighbor = graph[i][j];
        assertTrue("Graph neighbor index should be valid", neighbor >= 0 && neighbor < numVectors);
      }
    }
  }

  private static float[][] createSampleData() {
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };
  }

  private static float[][] createSampleQueries() {
    return new float[][] {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f}
    };
  }
}
