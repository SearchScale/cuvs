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
  SymbolLookup bridge;
  MemorySegment dataMS;

  private CagraIndex(CagraIndexParams params, float[][] dataset, Map<Integer, Integer> map, CuVSResources res)
      throws Throwable {
    this.mapping = map;
    this.params = params;
    this.dataset = dataset;
    this.init();
    this.res = res;
    this.ref = build();
  }

  private CagraIndex(InputStream in, CuVSResources res) throws Throwable {
    this.params = null;
    this.dataset = null;
    this.res = res;
    this.init();
    this.ref = deserialize(in);
  }

  private void init() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File wd = new File(System.getProperty("user.dir"));
    bridge = SymbolLookup.libraryLookup(wd.getParent() + "/api-sys/build/libcuvs_wrapper.so", arena);

    indexMH = linker.downcallHandle(bridge.findOrThrow("build_index"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS));

    searchMH = linker.downcallHandle(bridge.findOrThrow("search_index"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            linker.canonicalLayouts().get("int"), linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS));
  }

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

  private CagraIndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = dataset[0].length;
    ref = new CagraIndexReference(
        (MemorySegment) indexMH.invokeExact(getMemorySegment(dataset), rows, cols, res.resource));
    return ref;
  }

  public SearchResult search(CagraSearchParams params, float[][] queries) throws Throwable {

    MemorySegment rMS = (MemorySegment) searchMH.invokeExact(ref.indexMemorySegment, getMemorySegment(queries), 2, 4L,
        2L, res.resource);

    return null;
  }

  public void serialize(OutputStream out) {
  }

  private CagraIndexReference deserialize(InputStream in) {
    return null;
  }

  public CagraIndexParams getParams() {
    return params;
  }

  public PointerToDataset getDataset() {
    return null;
  }

  public CuVSResources getResources() {
    return res;
  }

  public static class Builder {
    private CagraIndexParams params;
    float[][] dataset;
    CuVSResources res;
    Map<Integer, Integer> map;

    InputStream in;

    public Builder(CuVSResources res) {
      this.res = res;
    }

    public Builder from(InputStream in) {
      this.in = in;
      return this;
    }

    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    public Builder withMapping(Map<Integer, Integer> map) {
      this.map = map;
      return this;
    }

    public Builder withIndexParams(CagraIndexParams params) {
      this.params = params;
      return this;
    }

    public CagraIndex build() throws Throwable {
      if (in != null) {
        return new CagraIndex(in, res);
      } else {
        return new CagraIndex(params, dataset, map, res);
      }
    }
  }

}
