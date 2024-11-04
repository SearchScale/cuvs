package ai.rapids.cuvs;

import ai.rapids.cuvs.cagra.*;
import ai.rapids.cuvs.cagra.CagraIndexParams.CuvsCagraGraphBuildAlgo;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import java.util.Map;

public class CagraIndexSearchTest {

	
    @Test
    public void testIndexingAndSearchingFlow() throws Throwable {
        // Sample data setup
        float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
                              { 0.6673192f, 0.10993068f } };
        Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
        float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
                              { 0.05198065f, 0.5789965f } };

        CuVSResources res = new CuVSResources();

        // Index and search parameter setup
        CagraIndexParams cagraIndexParams = new CagraIndexParams.Builder()
            .withIntermediateGraphDegree(10)
            .withBuildAlgo(CuvsCagraGraphBuildAlgo.IVF_PQ)
            .withWriterThreads(1)
            .build();
        System.out.println("Hello1");

        CagraSearchParams cagraSearchParams = new CagraSearchParams
            .Builder()
            .withMaxQueries(15)
            .build();
        System.out.println("Hello2");

        // Index creation
        CuVSIndex index = new CuVSIndex.Builder(res)
            .withDataset(dataset)
            .withANNAlgorithm(CuVSIndex.ANNAlgorithms.CAGRA)
            .withIndexParams(cagraIndexParams)
            .build();
        System.out.println("Hello3");

        // Query setup
        CuVSQuery query = new CuVSQuery.Builder()
            .withANNAlgorithm(CuVSIndex.ANNAlgorithms.CAGRA)
            .withSearchParams(cagraSearchParams)
            .withQueryVectors(queries)
            .withMapping(map)
            .build();
        System.out.println("Hello4");

        // Search
        SearchResult rslt = index.search(query);

        // Assert that results are not null
        assertNotNull(rslt, "SearchResult should not be null");
        assertNotNull(rslt.results, "SearchResult's results should not be null");
    }
}

