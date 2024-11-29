package com.nvidia.cuvs;

import java.lang.invoke.MethodHandles;
import java.util.Random;
import java.util.UUID;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.FileOutputStream;
import com.carrotsearch.randomizedtesting.RandomizedContext;
import org.junit.Ignore;


import static org.junit.Assert.assertEquals;

public class CagraRandomizedTest extends LuceneTestCase {
    private Random random;
    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @Before
    public void setup() {
        // Initialize the random instance and log the test seed for reproducibility
        this.random = random();
        log.info("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
    }
    @Ignore
    @Test
    public void testResultsTopKWithRandomValues() throws Throwable {
        // Generate a random dataset
        int numRows = random.nextInt(10) + 1; // 1 - 10 rows
        int numCols = random.nextInt(5) + 1;  // 1 - 5 columns
        float[][] dataset = new float[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                dataset[i][j] = random.nextFloat() * 100; // Random values between 0 and 100
            }
        }

        // Generate random query vectors
        int numQueries = random.nextInt(5) + 1; // 1 - 5 queries
        float[][] queries = new float[numQueries][numCols];
        for (int i = 0; i < numQueries; i++) {
            for (int j = 0; j < numCols; j++) {
                queries[i][j] = random.nextFloat() * 100; // Random values between 0 and 100
            }
        }

        // Set TopK to be within the range of dataset size
        int topK = random.nextInt(numRows) + 1;

        // Log dataset and query information for debugging
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

        // Create CuVSResources
        CuVSResources resources = new CuVSResources();

        // Create index parameters
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources).build();

        // Create the index
        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        // Create the query object
        CagraQuery query = new CagraQuery.Builder()
            .withQueryVectors(queries)
            .withTopK(topK)
            .withSearchParams(new CagraSearchParams.Builder(resources).build())
            .build();

        // Perform the search
        CagraSearchResults results = index.search(query);

        // Validate results
        results.getResults().forEach(result -> {
            log.info("Result size: {}", result.size());
            assertEquals("TopK mismatch for query.", Math.min(topK, numRows), result.size());
        });
    }

    @Test
    public void testSearchWithDeletedIndexFile() throws Throwable {
        Random random = random(); // Use LuceneTestCase random for reproducibility

        // Generate random dataset
        int numRows = random.nextInt(10) + 1; // 1 - 10 rows
        int numCols = random.nextInt(5) + 1;  // 1 - 5 columns
        float[][] dataset = new float[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                dataset[i][j] = random.nextFloat() * 100; // Random values between 0 and 100
            }
        }

        // Generate random query vectors
        int numQueries = random.nextInt(5) + 1; // 1 - 5 queries
        float[][] queries = new float[numQueries][numCols];
        for (int i = 0; i < numQueries; i++) {
            for (int j = 0; j < numCols; j++) {
                queries[i][j] = random.nextFloat() * 100; // Random values between 0 and 100
            }
        }

        // Set TopK value
        int topK = random.nextInt(numRows) + 1;

        // Log dataset and query details
        System.out.println("Dataset size: " + numRows + "x" + numCols);
        System.out.println("Query size: " + numQueries + "x" + numCols);
        System.out.println("TopK: " + topK);

        System.out.println("Dataset:");
        for (float[] row : dataset) {
            System.out.println(java.util.Arrays.toString(row));
        }

        System.out.println("Queries:");
        for (float[] query : queries) {
            System.out.println(java.util.Arrays.toString(query));
        }

        // Create resources and index parameters
        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        // Create and serialize the index
        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        String indexFileName = UUID.randomUUID().toString() + ".cag";
        index.serialize(new FileOutputStream(indexFileName));

        // Delete the serialized file
        File indexFile = new File(indexFileName);
        if (indexFile.exists()) {
            indexFile.delete();
        }

        // Attempt to create an InputStream from the deleted file
        Throwable exception = assertThrows(Exception.class, () -> {
            try (InputStream inputStream = new FileInputStream(indexFile)) {
                CagraIndex deletedIndex = new CagraIndex.Builder(resources)
                    .from(inputStream)
                    .build();

                CagraQuery query = new CagraQuery.Builder()
                    .withTopK(topK)
                    .withSearchParams(new CagraSearchParams.Builder(resources).build())
                    .withQueryVectors(queries)
                    .build();

                deletedIndex.search(query);
            }
        });

        // Assert the exception type
        assertTrue("Expected FileNotFoundException", exception instanceof java.io.FileNotFoundException);
    }

}
