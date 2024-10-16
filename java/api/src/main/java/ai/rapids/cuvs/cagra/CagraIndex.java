package ai.rapids.cuvs.cagra;

import java.io.File;
import java.io.InputStream;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment.Scope;
import java.lang.foreign.ValueLayout.OfBoolean;
import java.lang.foreign.ValueLayout.OfByte;
import java.lang.foreign.ValueLayout.OfChar;
import java.lang.foreign.ValueLayout.OfDouble;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.foreign.ValueLayout.OfShort;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Optional;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.stream.Stream;

import ai.rapids.cuvs.panama.cuvsCagraIndex;

public class CagraIndex {

  private CagraIndexParams params;
  private final float[][] dataset;
  private final CuVSResources res;
  private CagraIndexReference ref;

  Linker linker;
  Arena arena;
  MethodHandle indexMH;
  MethodHandle searchMH;
  SymbolLookup bridge;
  MemorySegment dataMS;
  MemorySegment resX;

  int rows;
  int cols;

  private void init() {

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
    
    resX = arena.allocate(ValueLayout.JAVA_LONG);

  }

  // Constructor that takes build params and dataset, creates an index
  private CagraIndex(CagraIndexParams params, float[][] dataset, CuVSResources res) throws Throwable {
    this.params = params;
    this.dataset = dataset;
    this.init();
    this.res = res;
    this.ref = build();
  }

  private CagraIndex(byte[] bytes, CuVSResources res) {
    this.params = null;
    this.dataset = null;
    this.res = res;
    this.init();
    this.ref = deserialize(bytes);
  }

  private CagraIndexReference build() throws Throwable {
    
    long rows = dataset.length;
    long cols = dataset[0].length;

    MemoryLayout dataML = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment dataMS = arena.allocate(dataML);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataML.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(dataMS, 0, 0, dataset[r][c]);
      }
    }
    
    // MemorySegment indexS = cuvsCagraIndex.allocate(arena);
    MemorySegment indexS = (MemorySegment)indexMH.invokeExact(dataMS, rows, cols, resX);
    ref = new CagraIndexReference(indexS);
    return ref;

  }

  public SearchResult search(CagraSearchParams params, float[][] queries) throws Throwable {

    long rows = queries.length;
    long cols = queries[0].length;

    MemoryLayout dataML = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment queryMS = arena.allocate(dataML);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataML.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(queryMS, 0, 0, queries[r][c]);
      }
    }
    
    searchMH.invokeExact(ref.indexS, queryMS, 2, 4L, 2L, resX);
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
