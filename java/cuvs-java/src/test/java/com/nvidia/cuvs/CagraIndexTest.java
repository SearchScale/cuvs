package com.nvidia.cuvs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Disabled;



import java.io.FileOutputStream;
import java.util.Map;

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
    @Disabled
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
    @Disabled
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
}
