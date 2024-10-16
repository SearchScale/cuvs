package ai.rapids.cuvs.cagra;

import java.lang.foreign.MemorySegment;

public class CagraIndexReference {

  public MemorySegment indexS;

  public CagraIndexReference(MemorySegment indexS) {
    super();
    this.indexS = indexS;
  }

}
