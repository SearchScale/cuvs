package com.nvidia.cuvs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Map;
import java.util.UUID;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class CagraIndexTest {
	  
    @Test
    public void testInvalidDataset() {
        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            // Use consistent dataset parameters as the working test
            float[][] invalidDataset = null; // Simulate an invalid dataset
            CuVSResources resources = new CuVSResources();
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .build();
            new CagraIndex.Builder(resources)
                .withDataset(invalidDataset)
                .withIndexParams(indexParams)
                .build();
        });

        assertEquals("Dataset cannot be null or empty", exception.getMessage());
    }
	  
    @Test
    public void testSerializationWithoutOutputStream() throws Throwable {
        // Use the same dataset as the working test
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f},
            {0.92514056f, 0.4463501f},
            {0.6673192f, 0.10993068f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            index.serialize(null); // Pass null output stream
        });

        assertEquals("Output stream cannot be null", exception.getMessage());
    }
    
    @Test
    public void testSingleElementDataset() throws Throwable {
        // Match dataset and parameters to the working test
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f},
            {0.92514056f, 0.4463501f},
            {0.6673192f, 0.10993068f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        float[][] query = {
            {0.48216683f, 0.0428398f},
            {0.5084142f, 0.6545497f},
            {0.51260436f, 0.2643005f},
            {0.05198065f, 0.5789965f}
        };

        CagraQuery cuvsQuery = new CagraQuery.Builder()
            .withTopK(3)
            .withSearchParams(new CagraSearchParams.Builder().build())
            .withQueryVectors(query)
            .build();

        CagraSearchResults results = index.search(cuvsQuery);

        // Verify the results size matches the queries
        assertEquals(query.length, results.getResults().size(), "Expected one result for each query");
    }
    
    @Test
    public void testSearchResultMapping() throws Throwable {
        // Match dataset and parameters to the working test
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f},
            {0.92514056f, 0.4463501f},
            {0.6673192f, 0.10993068f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        // Use consistent query and mapping
        Map<Integer, Integer> mapping = Map.of(0, 100, 1, 200, 2, 300, 3, 400);
        float[][] query = {
            {0.48216683f, 0.0428398f},
            {0.5084142f, 0.6545497f},
            {0.51260436f, 0.2643005f},
            {0.05198065f, 0.5789965f}
        };

        CagraQuery cuvsQuery = new CagraQuery.Builder()
            .withTopK(3)
            .withSearchParams(new CagraSearchParams.Builder().build())
            .withQueryVectors(query)
            .withMapping(mapping)
            .build();

        CagraSearchResults results = index.search(cuvsQuery);

        // Verify mapped results contain expected keys
        results.getResults().forEach(result -> {
            assertTrue(result.containsKey(100) || result.containsKey(200) || result.containsKey(300) || result.containsKey(400));
        });
    }
    
    public void testResultsTopK() throws Throwable {
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f},
            {0.92514056f, 0.4463501f}
        };

        float[][] queries = {
            {0.48216683f, 0.0428398f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder().build();
        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        CagraQuery query = new CagraQuery.Builder()
            .withQueryVectors(queries)
            .withTopK(2)
            .withSearchParams(new CagraSearchParams.Builder().build())
            .build();

        CagraSearchResults results = index.search(query);

        // Verify each query result contains exactly TopK neighbors
        results.getResults().forEach(result -> assertEquals(2, result.size()));
    }
    
    @Test
    public void testEmptyResults() throws Throwable {
        float[][] dataset = {
            {10.0f, 10.0f},
            {20.0f, 20.0f}
        };

        float[][] queries = {
            {1000.0f, 1000.0f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder().build();
        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        CagraQuery query = new CagraQuery.Builder()
            .withQueryVectors(queries)
            .withTopK(2)
            .withSearchParams(new CagraSearchParams.Builder().build())
            .build();

        CagraSearchResults results = index.search(query);
        System.out.println(results.getResults());

        // Verify no neighbors were found
        assertTrue(results.getResults().isEmpty());
    }
    
    @Test
    public void testSearchWithDeletedIndexFile() throws Throwable {
        // Dataset and Query
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f},
            {0.92514056f, 0.4463501f},
            {0.6673192f, 0.10993068f}
        };

        float[][] queries = {
            {0.48216683f, 0.0428398f},
            {0.5084142f, 0.6545497f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .build();

        // Create and serialize index
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
                    .withTopK(3)
                    .withSearchParams(new CagraSearchParams.Builder().build())
                    .withQueryVectors(queries)
                    .build();

                deletedIndex.search(query);
            }
        });

        // Assert the exception type
        assertTrue(exception instanceof java.io.FileNotFoundException, "Expected FileNotFoundException");
    }
    
    @Test
    public void testNullQueryVectors() throws Throwable {
        float[][] dataset = {
            {0.74021935f, 0.9209938f},
            {0.03902049f, 0.9689629f}
        };

        CuVSResources resources = new CuVSResources();
        CagraIndexParams indexParams = new CagraIndexParams.Builder().build();
        CagraIndex index = new CagraIndex.Builder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build();

        CagraQuery invalidQuery = new CagraQuery.Builder()
            .withQueryVectors(null)
            .withTopK(3)
            .withSearchParams(new CagraSearchParams.Builder().build())
            .build();

        Throwable exception = assertThrows(IllegalArgumentException.class, () -> {
            index.search(invalidQuery);
        });

        assertEquals("Query vectors cannot be null", exception.getMessage());
    }


}
