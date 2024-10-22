package ai.rapids.cuvs.cagra;

import java.lang.foreign.MemorySegment;

public class CagraIndexReference {

  public MemorySegment indexMemorySegment;

  /**
   * 
   * @param indexMemorySegment
   */
  public CagraIndexReference(MemorySegment indexMemorySegment) {
    super();
    this.indexMemorySegment = indexMemorySegment;
  }

}
