package ai.rapids.cuvs;

import ai.rapids.cuvs.cagra.*;
import ai.rapids.cuvs.cagra.CagraIndexParams.CuvsCagraGraphBuildAlgo;
import ai.rapids.cuvs.cagra.CuVSIndex.ANNAlgorithms;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Map;

public class CagraIndexSearchTest {

	
    @Test
    public void testIndexingAndSearchingFlow() throws Throwable {
        // Sample data setup
    	float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
    	        { 0.6673192f, 0.10993068f } };
    	    CuVSResources res = new CuVSResources();

    	    CagraIndexParams cagraIndexParams = new CagraIndexParams.Builder()
    	        .withBuildAlgo(CuvsCagraGraphBuildAlgo.NN_DESCENT)
    	        .build();
    	    System.out.println("Hello1");

    	    // Creating a new index
    	    CuVSIndex index = new CuVSIndex.Builder(res)
    	        .withDataset(dataset)
    	        .withANNAlgorithm(ANNAlgorithms.CAGRA)
    	        .withIndexParams(cagraIndexParams)
    	        .build();
    	    System.out.println("Hello3");

    }
}

