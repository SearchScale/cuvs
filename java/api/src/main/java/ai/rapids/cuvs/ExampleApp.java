package ai.rapids.cuvs;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Map;

import ai.rapids.cuvs.cagra.CagraIndex;
import ai.rapids.cuvs.cagra.CagraIndexParams;
import ai.rapids.cuvs.cagra.CagraSearchParams;
import ai.rapids.cuvs.cagra.CuVSResources;
import ai.rapids.cuvs.cagra.SearchResult;

public class ExampleApp {
  public static void main(String[] args) throws Throwable {

    float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f } };
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f } };

    CuVSResources res = new CuVSResources();


    // creating a new index
    CagraIndex index = new CagraIndex.Builder(res)
        .withDataset(dataset)
        .withMapping(map)
        .withIndexParams(new CagraIndexParams())
        .build();

    // saving the index on to the disk.
    index.serialize(new FileOutputStream("abc.cag"));
    
    // loading a cagra index from disk.
    InputStream fin = new FileInputStream(new File("abc.cag"));
    CagraIndex index2 = new CagraIndex.Builder(res)
        .from(fin)
        .build();
    
    SearchResult results = index.search(new CagraSearchParams(), queries);
    // System.out.println(results);

  }
}
