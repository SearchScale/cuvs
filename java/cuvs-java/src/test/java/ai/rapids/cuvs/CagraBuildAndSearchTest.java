package ai.rapids.cuvs;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import com.nvidia.cuvs.cagra.*;

public class CagraBuildAndSearchTest {

    private static final int VECTOR_DIMENSION = 2;
    private static final int DATASET_SIZE = 11;
    private static final int QUERY_SIZE = 4;
    private static final int TOP_K = 4;

    
    // Helper method to generate random vectors
    private float[][] generateRandomVectors(int size) {
        Random rand = new Random();
        float[][] vectors = new float[size][VECTOR_DIMENSION];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < VECTOR_DIMENSION; j++) {
                vectors[i][j] = rand.nextFloat(); // Random value between 0 and 1
            }
        }
        return vectors;
    }

    // Helper method to calculate Euclidean distance
    private double calculateDistance(float[] v1, float[] v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sum += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sum);
    }

    // Helper method to calculate top-K distances manually
    private List<Map.Entry<Integer, Double>> getTopKDistances(float[] query, float[][] dataset, int k) {
        PriorityQueue<Map.Entry<Integer, Double>> heap = new PriorityQueue<>(
            Comparator.<Map.Entry<Integer, Double>>comparingDouble(Map.Entry::getValue).reversed()
        );

        for (int i = 0; i < dataset.length; i++) {
            double distance = calculateDistance(query, dataset[i]);
            if (heap.size() < k) {
                heap.offer(new AbstractMap.SimpleEntry<>(i, distance));
            } else if (distance < heap.peek().getValue()) {
                heap.poll();
                heap.offer(new AbstractMap.SimpleEntry<>(i, distance));
            }
        }

        return heap.stream()
                   .sorted(Comparator.comparingDouble(Map.Entry::getValue))
                   .collect(Collectors.toList());
    }

    @Test
    public void testIndexingAndSearchingFlow() throws Throwable {
        
        float[][] dataset = generateRandomVectors(DATASET_SIZE);
        float[][] queries = generateRandomVectors(QUERY_SIZE);

        // Create map for dataset IDs
        Map<Integer, Integer> map = IntStream.range(0, dataset.length)
                                             .boxed()
                                             .collect(Collectors.toMap(i -> i, i -> i));

        CuVSResources res = new CuVSResources();

        // Configure index parameters
        CagraIndexParams cagraIndexParams = new CagraIndexParams.Builder()
                .withIntermediateGraphDegree(10)
                .withBuildAlgo(CagraIndexParams.CuvsCagraGraphBuildAlgo.IVF_PQ)
                .withWriterThreads(1)
                .build();

        // Create the index with the dataset
        CagraIndex index = new CagraIndex.Builder(res)
                .withDataset(dataset)
                .withIndexParams(cagraIndexParams)
                .build();

        // Configure search parameters
        CagraSearchParams cagraSearchParams = new CagraSearchParams.Builder().build();

        // Create a query object with the query vectors
        CuVSQuery query = new CuVSQuery.Builder()
                .withTopK(TOP_K) // Retrieve top-K neighbors
                .withSearchParams(cagraSearchParams)
                .withQueryVectors(queries)
                .withMapping(map)
                .build();

        // Perform the search
        SearchResult rslt = index.search(query);

        // Ensure the search results are not null
        assertNotNull(rslt, "SearchResult instance should not be null.");
        assertNotNull(rslt.getResults(), "Search results should not be null.");

        // Validate the results for each query
        for (int queryIndex = 0; queryIndex < QUERY_SIZE; queryIndex++) {
            
            Map<Integer, Float> queryResults = rslt.getResults();
            assertNotNull(queryResults, "Results for query " + queryIndex + " should not be null.");

            // Calculate expected top-K distances
            List<Map.Entry<Integer, Double>> expectedTopK = getTopKDistances(queries[queryIndex], dataset, TOP_K);

            // printing expected and actual results
            System.out.println("\nQuery " + queryIndex + ":");
            System.out.println("Expected Top-K Results:");
            expectedTopK.forEach(entry ->
                System.out.println("Index: " + entry.getKey() + ", Distance: " + entry.getValue()));

            System.out.println("Actual Top-K Results:");
            queryResults.entrySet().stream()
                        .sorted(Map.Entry.comparingByValue())
                        .limit(TOP_K)
                        .forEach(entry ->
                            System.out.println("Index: " + entry.getKey() + ", Distance: " + entry.getValue()));
            assertEquals(TOP_K, queryResults.size(), "Number of results does not match for query " + queryIndex);
            assertEquals(expectedTopK.size(), TOP_K, "Number of expected top-K vectors does not match TOP_K.");
            assertEquals(queryResults.size(), expectedTopK.size(), "Number of actual results does not match the number of expected results.");

            // Validate each result
            /*for (int i = 0; i < TOP_K; i++) {
                Integer expectedIndex = expectedTopK.get(i).getKey();
                double expectedDistance = expectedTopK.get(i).getValue();

                // Assert key exists
                assertTrue(queryResults.containsKey(expectedIndex),
                           "Expected index " + expectedIndex + " not found in query results for query " + queryIndex);

                // Validate distances
                float actualDistance = queryResults.get(expectedIndex);
                assertEquals(expectedDistance, actualDistance, 0.001,
                             "Distances do not match for query " + queryIndex + " at position " + i);
            }*/
        }
    }

}
