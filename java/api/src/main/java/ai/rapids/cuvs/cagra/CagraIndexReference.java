package ai.rapids.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import ai.rapids.cuvs.panama.cuvsCagraIndex;

public class CagraIndexReference {

  public MemorySegment indexMemorySegment;
  
  /**
   * 
   */
  public CagraIndexReference() {
    Arena arena = Arena.ofConfined();
    indexMemorySegment = cuvsCagraIndex.allocate(arena);
  }

  /**
   * 
   * @param indexMemorySegment
   */
  public CagraIndexReference(MemorySegment indexMemorySegment) {
    this.indexMemorySegment = indexMemorySegment;
  }

}
