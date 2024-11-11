package ai.rapids.cuvs;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

import org.junit.jupiter.api.Test;

import com.nvidia.cuvs.cagra.CagraIndex;
import com.nvidia.cuvs.cagra.CagraIndexParams;
import com.nvidia.cuvs.cagra.CagraIndexParams.CuvsCagraGraphBuildAlgo;
import com.nvidia.cuvs.cagra.CagraSearchParams;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.CuVSResources;
import com.nvidia.cuvs.cagra.SearchResult;

public class CagraBuildAndSearchTest {

	@Test
	public void testIndexingAndSearchingFlow() throws Throwable {
		// Sample data setup
		float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
				{ 0.6673192f, 0.10993068f } };
		Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
		float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
				{ 0.05198065f, 0.5789965f } };

		CuVSResources res = new CuVSResources();

		CagraIndexParams cagraIndexParams = new CagraIndexParams.Builder()
				.withIntermediateGraphDegree(10)
				.withBuildAlgo(CuvsCagraGraphBuildAlgo.IVF_PQ)
				.withWriterThreads(1)
				.build();

		// Creating a new index
		CagraIndex index = new CagraIndex.Builder(res)
				.withDataset(dataset)
				.withIndexParams(cagraIndexParams)
				.build();

		// Search
		CagraSearchParams cagraSearchParams = new CagraSearchParams
				.Builder()
				.build();

		CuVSQuery query = new CuVSQuery.Builder()
				.withSearchParams(cagraSearchParams)
				.withQueryVectors(queries)
				.withMapping(map)
				.build();

		SearchResult rslt = index.search(query);

		// Assert that 2 results are returned
		assertEquals(2, rslt.getResults().size(), "2 results should be returned.");
	}

}
