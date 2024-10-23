package ai.rapids.cuvs.panama;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.stream.Collectors;

public class results_h {

  results_h() {
    // Should not be called directly
  }

  static final Arena LIBRARY_ARENA = Arena.ofAuto();
  static final boolean TRACE_DOWNCALLS = Boolean.getBoolean("jextract.trace.downcalls");

  static void traceDowncall(String name, Object... args) {
    String traceArgs = Arrays.stream(args).map(Object::toString).collect(Collectors.joining(", "));
    System.out.printf("%s(%s)\n", name, traceArgs);
  }

  static MemorySegment findOrThrow(String symbol) {
    return SYMBOL_LOOKUP.find(symbol).orElseThrow(() -> new UnsatisfiedLinkError("unresolved symbol: " + symbol));
  }

  static MethodHandle upcallHandle(Class<?> fi, String name, FunctionDescriptor fdesc) {
    try {
      return MethodHandles.lookup().findVirtual(fi, name, fdesc.toMethodType());
    } catch (ReflectiveOperationException ex) {
      throw new AssertionError(ex);
    }
  }

  static MemoryLayout align(MemoryLayout layout, long align) {
    return switch (layout) {
    case PaddingLayout p -> p;
    case ValueLayout v -> v.withByteAlignment(align);
    case GroupLayout g -> {
      MemoryLayout[] alignedMembers = g.memberLayouts().stream().map(m -> align(m, align)).toArray(MemoryLayout[]::new);
      yield g instanceof StructLayout ? MemoryLayout.structLayout(alignedMembers)
          : MemoryLayout.unionLayout(alignedMembers);
    }
    case SequenceLayout s -> MemoryLayout.sequenceLayout(s.elementCount(), align(s.elementLayout(), align));
    };
  }

  static final SymbolLookup SYMBOL_LOOKUP = SymbolLookup.loaderLookup().or(Linker.nativeLinker().defaultLookup());

  public static final ValueLayout.OfBoolean C_BOOL = ValueLayout.JAVA_BOOLEAN;
  public static final ValueLayout.OfByte C_CHAR = ValueLayout.JAVA_BYTE;
  public static final ValueLayout.OfShort C_SHORT = ValueLayout.JAVA_SHORT;
  public static final ValueLayout.OfInt C_INT = ValueLayout.JAVA_INT;
  public static final ValueLayout.OfLong C_LONG_LONG = ValueLayout.JAVA_LONG;
  public static final ValueLayout.OfFloat C_FLOAT = ValueLayout.JAVA_FLOAT;
  public static final ValueLayout.OfDouble C_DOUBLE = ValueLayout.JAVA_DOUBLE;
  public static final AddressLayout C_POINTER = ValueLayout.ADDRESS
      .withTargetLayout(MemoryLayout.sequenceLayout(java.lang.Long.MAX_VALUE, JAVA_BYTE));
  public static final ValueLayout.OfLong C_LONG = ValueLayout.JAVA_LONG;
  /**
   * {@snippet lang = c :
   * typedef struct results {
   *     int test;
   *     int *neighbors_h;
   *     float *distances_h;
   * } *rsltp
   * }
   */
  public static final AddressLayout rsltp = results_h.C_POINTER;
}