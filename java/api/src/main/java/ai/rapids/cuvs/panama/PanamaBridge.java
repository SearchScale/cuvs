package ai.rapids.cuvs.panama;

import java.io.File;
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

public class PanamaBridge {

  Linker linker;
  Arena arena;
  SymbolLookup bridge;
  MemorySegment res;
  MemorySegment fln;
  MethodHandle search_index;

  public PanamaBridge() throws Throwable {

    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();
    res = arena.allocate(ValueLayout.JAVA_LONG);

    File wd = new File(System.getProperty("user.dir"));
    bridge = SymbolLookup.libraryLookup(wd.getParent() + "/cuda/build/libdemo.so", arena);

    search_index = linker.downcallHandle(bridge.findOrThrow("search_index"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"), linker.canonicalLayouts().get("bool")));

  }

  private class DataAndShapeWrapper {
    public MemorySegment dataMS;
    public MemorySegment shapeMS;

    public DataAndShapeWrapper(MemorySegment dataMS, MemorySegment shapeMS) {
      super();
      this.dataMS = dataMS;
      this.shapeMS = shapeMS;
    }
  }

  public DataAndShapeWrapper getDataMemorySegment(boolean empty, float[][] data, int rows, int cols) {

    if (!empty) {
      rows = data.length;
      cols = data[0].length;
    }

    MemoryLayout dataML = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment dataMS = arena.allocate(dataML);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataML.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(dataMS, 0, 0, empty ? 0 : data[r][c]);
      }
    }

    MemoryLayout shapeML = MemoryLayout.sequenceLayout(2, linker.canonicalLayouts().get("int64_t"));
    MemorySegment shapeMS = arena.allocate(shapeML);
    VarHandle element0 = shapeML.varHandle(PathElement.sequenceElement(0));
    element0.set(shapeMS, 0, rows);
    VarHandle element1 = shapeML.varHandle(PathElement.sequenceElement(1));
    element1.set(shapeMS, 0, cols);

    return new DataAndShapeWrapper(dataMS, shapeMS);
  }

  public MemorySegment wrapTensor(DataAndShapeWrapper wrapper, int code) {

    MemorySegment dlDataType = DLDataType.allocate(arena);
    DLDataType.code(dlDataType, (byte) code);
    DLDataType.bits(dlDataType, (byte) 32);
    DLDataType.lanes(dlDataType, (byte) 1);

    MemorySegment dlDeviceType = DLDevice.allocate(arena);
    DLDevice.device_type(dlDeviceType, 2);

    MemorySegment dlTensor = DLTensor.allocate(arena);
    DLTensor.shape(dlTensor, wrapper.shapeMS);
    DLTensor.data(dlTensor, wrapper.dataMS);
    DLTensor.ndim(dlTensor, 2);
    DLTensor.device(dlTensor, dlDeviceType);
    DLTensor.dtype(dlTensor, dlDataType);

    return dlTensor;
  }

  /**
   * Java String -> C char*
   * 
   * @param str
   * @return MemorySegment
   */
  public MemorySegment getStringSegment(StringBuilder str) {
    str.append('\0');
    MemoryLayout sq = MemoryLayout.sequenceLayout(str.length(), linker.canonicalLayouts().get("char"));
    MemorySegment fln = arena.allocate(sq);

    for (int i = 0; i < str.length(); i++) {
      VarHandle flnVH = sq.varHandle(PathElement.sequenceElement(i));
      flnVH.set(fln, 0L, (byte) str.charAt(i));
    }
    return fln;
  }

  public void createIndex(MemorySegment datasetTensor, MemorySegment index) throws Throwable {
    MemoryLayout cuvsCagraIndexParamsML = ValueLayout.ADDRESS.withTargetLayout(cuvsCagraIndexParams.layout());
    MemorySegment cuvsCagraIndexParamsMS = arena.allocate(cuvsCagraIndexParamsML);

    MethodHandle cuvs_cagra_index_params_create = linker.downcallHandle(
        bridge.findOrThrow("cuvs_cagra_index_params_create"), FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    cuvs_cagra_index_params_create.invokeExact(cuvsCagraIndexParamsMS);

    MethodHandle cuvs_cagra_build = linker.downcallHandle(bridge.findOrThrow("cuvs_cagra_build"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    cuvs_cagra_build.invokeExact(res, cuvsCagraIndexParamsMS, datasetTensor, index);
  }

  public void serializeIndex(MemorySegment res, MemorySegment index, MemorySegment filename, boolean include_dataset)
      throws Throwable {
    MethodHandle cuvs_cagra_serialize = linker.downcallHandle(bridge.findOrThrow("cuvs_cagra_serialize"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            linker.canonicalLayouts().get("bool")));
    cuvs_cagra_serialize.invokeExact(res, filename, index, include_dataset);
  }

  public void deserializeIndex(MemorySegment res, MemorySegment index, MemorySegment filename) throws Throwable {
    MethodHandle cuvs_cagra_deserialize = linker.downcallHandle(bridge.findOrThrow("cuvs_cagra_deserialize"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    cuvs_cagra_deserialize.invokeExact(res, filename, index);
  }

  public void test() throws Throwable {

    int topk = 2;
    boolean debug = true;
    MemorySegment fln = this.getStringSegment(new StringBuilder("trial_segment.cag"));
    float[][] dataset = { { 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f },
        { 0.6673192f, 0.10993068f } };
    float[][] queries = { { 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f },
        { 0.05198065f, 0.5789965f } };

    MemorySegment index = cuvsCagraIndex.allocate(arena);

    this.createIndex(this.wrapTensor(this.getDataMemorySegment(false, dataset, 0, 0), 2), index);
    
    // this.serializeIndex(res, index, fln, true);
    // this.deserializeIndex(res, index, fln);

    MemorySegment queriesTensor = this.wrapTensor(this.getDataMemorySegment(false, queries, 0, 0), 2);
    MemorySegment neighborsTensor = this.wrapTensor(this.getDataMemorySegment(true, null, queries.length, topk), 1);
    MemorySegment distanceTensor = this.wrapTensor(this.getDataMemorySegment(true, null, queries.length, topk), 2);

    search_index.invokeExact(res, index, queriesTensor, neighborsTensor, distanceTensor, topk, debug);

    System.out.println("**************************************");
    MemoryLayout neighborsML = MemoryLayout.sequenceLayout(queries.length,
        MemoryLayout.sequenceLayout(topk, linker.canonicalLayouts().get("int")));
    MemoryLayout distancesML = MemoryLayout.sequenceLayout(queries.length,
        MemoryLayout.sequenceLayout(topk, linker.canonicalLayouts().get("float")));

    MemoryLayout neighborsMLP = DLTensor.data$layout().withTargetLayout(neighborsML);
    MemoryLayout distancesMLP = DLTensor.data$layout().withTargetLayout(distancesML);

    for (int r = 0; r < 1; r++) {
      for (int c = 0; c < topk; c++) {
        VarHandle nelement = neighborsMLP.arrayElementVarHandle(PathElement.dereferenceElement(),
            PathElement.sequenceElement(r), PathElement.sequenceElement(c));
        VarHandle delement = distancesMLP.arrayElementVarHandle(PathElement.dereferenceElement(),
            PathElement.sequenceElement(r), PathElement.sequenceElement(c));
        System.out.println("Neighbor: " + nelement.get(neighborsTensor, 0L, 0L) + ", Distance: "
            + delement.get(distanceTensor, 0L, 0L));
      }
    }
    System.out.println("**************************************");
  }
}
