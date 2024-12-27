package com.nvidia.cuvs;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.hnsw.HnswQuery;
import com.nvidia.cuvs.hnsw.HnswSearchParameters;
import com.nvidia.cuvs.hnsw.HnswSearchResults;
import com.nvidia.cuvs.hnsw.HnswUtil;

public class TestCagraToHnsw {

  private static final Logger log = LoggerFactory.getLogger(TestCagraToHnsw.class);

  @Test
  public void testSerializationAndSearch() throws Throwable {
    String cagraFilePath = "cagra_index.bin";
    String hnswFilePath = "hnsw_index.bin";

    try (CuVSResources resources = new CuVSResources()) {
      log.info("Starting CAGRA to HNSW test.");

      // Step 1: Build a sample dataset
      float[][] dataset = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f } };

      log.info("Building CAGRA index...");
      CagraIndex cagraIndex = new CagraIndex.Builder(resources).withDataset(dataset)
          .withIndexParams(new CagraIndexParams.Builder(resources).build()).build();
      log.info("CAGRA index built successfully.");

      log.info("Serializing CAGRA index to file: {}", cagraFilePath);
      try (FileOutputStream outputStream = new FileOutputStream(cagraFilePath)) {
        cagraIndex.serialize(outputStream);
      }
      assertTrue("CAGRA index file should exist", new File(cagraFilePath).exists());
      log.info("CAGRA index serialized to: {}", cagraFilePath);

      log.info("Converting CAGRA index to HNSW format...");
      HnswUtil.serializeCagraToHnsw(resources, cagraFilePath, hnswFilePath);

      File hnswFile = new File(hnswFilePath);
      assertTrue("HNSW index file should exist", hnswFile.exists());
      log.info("HNSW index file successfully created at: {}", hnswFilePath);

      // Step 2: Prepare a memory segment for the HNSW index
      MemorySegment hnswIndexSegment = resources.arena.allocate(ValueLayout.ADDRESS.byteSize());

      // Step 3: Perform a search
      log.info("Starting HNSW search...");
      float[][] queryVectors = { { 2.0f, 3.0f, 4.0f }, { 5.0f, 6.0f, 7.0f } };
      int topK = 2;

      // Create search parameters
      HnswSearchParameters searchParams = new HnswSearchParameters(20, 2);

      // Create query object with the correct constructor
      HnswQuery query = new HnswQuery(hnswIndexSegment, queryVectors, topK, searchParams);

      // Perform the search
      HnswSearchResults results = HnswUtil.search(resources, query);

      // Validate results
      assertNotNull("Search results should not be null", results);
      assertTrue("Search results should have neighbors", results.getNeighbors().length > 0);
      assertTrue("Search results should have distances", results.getDistances().length > 0);
      log.info("Search results retrieved: Neighbors = {}, Distances = {}", results.getNeighbors(),
          results.getDistances());

      Files.deleteIfExists(Path.of(cagraFilePath));
      Files.deleteIfExists(Path.of(hnswFilePath));
      log.info("Test completed successfully. Temporary files cleaned up.");
    } catch (Exception e) {
      log.error("Test failed: ", e);
      throw e; // Re-throw exception to mark test as failed
    }
  }
}
