package ai.rapids.cuvs.cagra;

import java.io.File;
import java.io.InputStream;
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

public class CagraIndex {

  private CagraIndexParams params;
  private final float[][] dataset;
  private final CuVSResources res;
  private CagraIndexReference ref;

  Linker linker;
  Arena arena;
  MethodHandle cresMH;
  MethodHandle indexMH;
  MethodHandle searchMH;
  SymbolLookup bridge;
  MemorySegment dataMS;

  private CagraIndex(CagraIndexParams params, float[][] dataset, CuVSResources res) throws Throwable {
    this.params = params;
    this.dataset = dataset;
    this.init();
    this.res = res;
    this.ref = build();
  }

  private CagraIndex(byte[] bytes, CuVSResources res) throws Throwable {
    this.params = null;
    this.dataset = null;
    this.res = res;
    this.init();
    this.ref = deserialize(bytes);
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
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS));
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
    searchMH.invokeExact(ref.indexMemorySegment, getMemorySegment(queries), 2, 4L, 2L, res.resource);
    return null;
  }

  public InputStream serialize() {
    return null;
  }

  private CagraIndexReference deserialize(byte[] bytes) {
    return null;
  }

  public CagraIndexParams getParams() {
    return params;
  }

  public float[][] getDataset() {
    return dataset;
  }

  public CuVSResources getResources() {
    return res;
  }

  public static class Builder {
    private CagraIndexParams params;
    float[][] dataset;
    CuVSResources res;

    byte[] bytes;

    public Builder(CuVSResources res) {
      this.res = res;
    }

    public Builder fromBytes(byte[] bytes) {
      this.bytes = bytes;
      return this;
    }

    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    public Builder withIndexParams(CagraIndexParams params) {
      this.params = params;
      return this;
    }

    public CagraIndex build() throws Throwable {
      if (bytes != null) {
        return new CagraIndex(bytes, res);
      } else {
        return new CagraIndex(params, dataset, res);
      }
    }
  }

}
