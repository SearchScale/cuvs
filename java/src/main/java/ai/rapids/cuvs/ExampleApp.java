package ai.rapids.cuvs;

import ai.rapids.cuvs.cagra.CagraIndex;
import ai.rapids.cuvs.cagra.CagraIndexParams;
import ai.rapids.cuvs.cagra.CagraSearchParams;
import ai.rapids.cuvs.cagra.CuVSResources;
import ai.rapids.cuvs.cagra.SearchResult;

public class ExampleApp {
    public static void main(String[] args) {
    	CuVSResources res = new CuVSResources();
    	
        CagraIndex index = new CagraIndex.Builder(res)
        		.withDataset(new float[][]{{0f, 0f}, {1f, 1f}})
        		.withIndexParams(new CagraIndexParams())
        		.build();
        
        index.serialize();
    	
    	/*byte[] bytes = null ; // load a .cag file from disk
    	CagraIndex index = new CagraIndex.Builder(res)
    			.fromBytes(bytes)
    			.build();*/
        
        SearchResult results = index.search(new CagraSearchParams(), new float[][]{{0.25f, 0.2f}});
        System.out.println(results);
    }
}
