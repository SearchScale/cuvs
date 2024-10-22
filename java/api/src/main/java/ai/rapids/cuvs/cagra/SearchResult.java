package ai.rapids.cuvs.cagra;

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SearchResult {

  public List<int[]> neighbours;
  public List<float[][]> distances;
  public Map<Integer, Float> results;
  SequenceLayout neighboursSL;
  SequenceLayout distancesSL;
  MemorySegment neighboursMS;
  MemorySegment distancesMS;
  int topK;

  public SearchResult(SequenceLayout neighboursSL, SequenceLayout distancesSL, MemorySegment neighboursMS,
      MemorySegment distancesMS, int topK) {
    super();
    this.topK = topK;
    this.neighboursSL = neighboursSL;
    this.distancesSL = distancesSL;
    this.neighboursMS = neighboursMS;
    this.distancesMS = distancesMS;
    neighbours = new ArrayList<int[]>();
    distances = new ArrayList<float[][]>();
    results = new HashMap<Integer, Float>();
    this.load();
  }

  private void load() {
    VarHandle neighboursVH = neighboursSL.varHandle(PathElement.sequenceElement());
    VarHandle distancesVH = distancesSL.varHandle(PathElement.sequenceElement());

    for (long i = 0; i < topK; i++) {
      results.put((int) neighboursVH.get(neighboursMS, 0L, i), (float) distancesVH.get(distancesMS, 0L, i));
    }
  }

}