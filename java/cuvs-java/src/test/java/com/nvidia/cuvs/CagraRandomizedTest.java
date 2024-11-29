package com.nvidia.cuvs;

import java.lang.invoke.MethodHandles;
import java.util.Map;
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
        
        this.random = random();
        log.info("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
    }
    @Ignore
    @Test
    public void testInvalidDataset() throws Throwable {
        float[][] invalidDataset = null; // Simulate an invalid dataset

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            new CagraIndex.Builder(resources)
                .withDataset(invalidDataset)
                .withIndexParams(indexParams)
                .build();
        });

        assertEquals("Dataset cannot be null or empty", exception.getMessage());
    }
    @Ignore
    @Test
    public void testSerializationWithoutOutputStream() throws Throwable {
        // Randomize dataset
        int numRows = random.nextInt(10) + 1;
        int numCols = random.nextInt(5) + 1;
        float[][] dataset = new float[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                dataset[i][j] = random.nextFloat() * 100;
            }
        }

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            index.serialize(null);
        });

        assertEquals("Output stream cannot be null", exception.getMessage());
    }
    
    @Test
    public void testSearchResultMapping() throws Throwable {
        // Randomize dataset
        int numRows = random.nextInt(10) + 1; 
        int numCols = random.nextInt(5) + 1;
        float[][] dataset = new float[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                dataset[i][j] = random.nextFloat() * 100;
            }
        }

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        
        Map<Integer, Integer> mapping = new java.util.HashMap<>();
        for (int i = 0; i < numRows; i++) {
            mapping.put(i, i + 1000);
        }

        // Randomize query vectors
        float[][] query = new float[4][numCols];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < numCols; j++) {
                query[i][j] = random.nextFloat() * 100;
            }
        }

        CagraQuery cuvsQuery = new CagraQuery.Builder()
            .withTopK(3)
            .withSearchParams(new CagraSearchParams.Builder(resources).build())
            .withQueryVectors(query)
            .withMapping(mapping)
            .build();

        CagraSearchResults results = index.search(cuvsQuery);

        // Validate the results
        results.getResults().forEach(result -> {
            result.keySet().forEach(key -> {
                assertNotNull("Key should not be null", key);
                assertTrue("Key not in mapping: " + key, mapping.containsValue(key));
            });
        });
    }


    @Ignore
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

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        CagraQuery query = new CagraQuery.Builder()
            .withQueryVectors(queries)
            .withTopK(topK)
            .withSearchParams(new CagraSearchParams.Builder(resources).build())
            .build();

        CagraSearchResults results = index.search(query);

        results.getResults().forEach(result -> {
            log.info("Result size: {}", result.size());
            assertEquals("TopK mismatch for query.", Math.min(topK, numRows), result.size());
        });
    }
    @Ignore
    @Test
    public void testSearchWithDeletedIndexFile() throws Throwable {
        Random random = random();

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

        int topK = random.nextInt(numRows) + 1;

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

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder(resources)
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        String indexFileName = UUID.randomUUID().toString() + ".cag";
        index.serialize(new FileOutputStream(indexFileName));

        File indexFile = new File(indexFileName);
        if (indexFile.exists()) {
            indexFile.delete();
        }

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

        assertTrue("Expected FileNotFoundException", exception instanceof java.io.FileNotFoundException);
    }

}
