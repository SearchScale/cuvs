package ai.rapids.cuvs.cagra;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.Map;

public class CagraIndex {

  private CagraIndexParams params;
  private final float[][] dataset;
  private final CuVSResources res;
  private CagraIndexReference ref;

  private Map<Integer, Integer> mapping; // nocommit (this should be int[], not a mapping)

  Linker linker;
  Arena arena;
  MethodHandle cresMH;
  MethodHandle indexMH;
  MethodHandle searchMH;
  MethodHandle testMH;
  SymbolLookup bridge;
  MemorySegment dataMS;

  /**
   * 
   * @param params
   * @param dataset
   * @param map
   * @param res
   * @throws Throwable
   */
  private CagraIndex(CagraIndexParams params, float[][] dataset, Map<Integer, Integer> map, CuVSResources res)
      throws Throwable {
    this.mapping = map;
    this.params = params;
    this.dataset = dataset;
    this.init();
    this.res = res;
    this.ref = build();
  }

  /**
   * 
   * @param in
   * @param res
   * @throws Throwable
   */
  private CagraIndex(InputStream in, CuVSResources res) throws Throwable {
    this.params = null;
    this.dataset = null;
    this.res = res;
    this.init();
    this.ref = deserialize(in);
  }

  /**
   * 
   * @throws Throwable
   */
  private void init() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File wd = new File(System.getProperty("user.dir"));
    bridge = SymbolLookup.libraryLookup(wd.getParent() + "/api-sys/build/libcuvs_wrapper.so", arena);

    indexMH = linker.downcallHandle(bridge.findOrThrow("build_index"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMH = linker.downcallHandle(bridge.findOrThrow("search_index"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  }

  /**
   * 
   * @param data
   * @return
   */
  private MemorySegment getMemorySegment(float[][] data) {
    long rows = data.length;
    long cols = data[0].length;

    MemoryLayout dataML = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment dataMS = arena.allocate(dataML);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataML.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(dataMS, 0, 0, data[r][c]);
      }
    }

    return dataMS;
  }

  /**
   * 
   * @return
   * @throws Throwable
   */
  private CagraIndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = dataset[0].length;
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);

    ref = new CagraIndexReference(
        (MemorySegment) indexMH.invokeExact(getMemorySegment(dataset), rows, cols, res.resource, rvMS, params.cagraIndexParamsMS));

    System.out.println("Build call return value: " + rvMS.get(ValueLayout.JAVA_INT, 0));

    return ref;
  }

  /**
   * 
   * @param params
   * @param queries
   * @return
   * @throws Throwable
   */
  public SearchResult search(CagraSearchParams params, float[][] queries) throws Throwable {

    SequenceLayout neighborsSL = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("int"));
    SequenceLayout distancesSL = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("float"));
    MemorySegment neighborsMS = arena.allocate(neighborsSL);
    MemorySegment distancesMS = arena.allocate(distancesSL);
    MemoryLayout rvML = linker.canonicalLayouts().get("int");
    MemorySegment rvMS = arena.allocate(rvML);

    searchMH.invokeExact(ref.indexMemorySegment, getMemorySegment(queries), 2, 4L, 2L, res.resource, neighborsMS,
        distancesMS, rvMS);

    System.out.println("Search call return value: " + rvMS.get(ValueLayout.JAVA_INT, 0));

    return new SearchResult(neighborsSL, distancesSL, neighborsMS, distancesMS, 2);
  }

  /**
   * 
   * @param out
   */
  public void serialize(OutputStream out) {
  }

  /**
   * 
   * @param in
   * @return
   */
  private CagraIndexReference deserialize(InputStream in) {
    return null;
  }

  /**
   * 
   * @return
   */
  public CagraIndexParams getParams() {
    return params;
  }

  /**
   * 
   * @return
   */
  public PointerToDataset getDataset() {
    return null;
  }

  /**
   * 
   * @return
   */
  public CuVSResources getResources() {
    return res;
  }

  public static class Builder {
    private CagraIndexParams params;
    float[][] dataset;
    CuVSResources res;
    Map<Integer, Integer> map;

    InputStream in;

    /**
     * 
     * @param res
     */
    public Builder(CuVSResources res) {
      this.res = res;
    }

    /**
     * 
     * @param in
     * @return
     */
    public Builder from(InputStream in) {
      this.in = in;
      return this;
    }

    /**
     * 
     * @param dataset
     * @return
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * 
     * @param map
     * @return
     */
    public Builder withMapping(Map<Integer, Integer> map) {
      this.map = map;
      return this;
    }

    /**
     * 
     * @param params
     * @return
     */
    public Builder withIndexParams(CagraIndexParams params) {
      this.params = params;
      return this;
    }

    /**
     * 
     * @return
     * @throws Throwable
     */
    public CagraIndex build() throws Throwable {
      if (in != null) {
        return new CagraIndex(in, res);
      } else {
        return new CagraIndex(params, dataset, map, res);
      }
    }
  }

}
