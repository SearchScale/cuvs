package ai.rapids.cuvs.cagra;

import java.lang.foreign.MemorySegment;

public class CagraIndexReference {

  public MemorySegment indexMemorySegment;

  public CagraIndexReference(MemorySegment indexMemorySegment) {
    super();
    this.indexMemorySegment = indexMemorySegment;
  }

}
