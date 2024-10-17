package ai.rapids.cuvs.cagra;

import java.io.File;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class CuVSResources {

  private Linker linker;
  private Arena arena;
  private MethodHandle cresMH;
  private SymbolLookup bridge;
  public MemorySegment resource;

  public CuVSResources() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File wd = new File(System.getProperty("user.dir"));
    bridge = SymbolLookup.libraryLookup(wd.getParent() + "/api-sys/build/libcuvs_wrapper.so", arena);

    cresMH = linker.downcallHandle(bridge.findOrThrow("create_resource"), FunctionDescriptor.of(ValueLayout.ADDRESS));
    resource = (MemorySegment) cresMH.invokeExact();
  }

}
