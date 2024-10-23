package ai.rapids.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import ai.rapids.cuvs.panama.cuvsCagraIndexParams;

/*
* struct cuvsCagraIndexParams {
*     size_t intermediate_graph_degree;
*     size_t graph_degree;
*     enum cuvsCagraGraphBuildAlgo build_algo;
*     size_t nn_descent_niter;
* }
*/
public class CagraIndexParams {

  public enum CuvsCagraGraphBuildAlgo {
    AUTO_SELECT(0), IVF_PQ(1), NN_DESCENT(2);

    public final int label;

    private CuvsCagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  public MemorySegment cagraIndexParamsMS;

  public CagraIndexParams(Arena arena, int igd, int gd, CuvsCagraGraphBuildAlgo ba, int nndn) {
    cagraIndexParamsMS = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(cagraIndexParamsMS, igd);
    cuvsCagraIndexParams.graph_degree(cagraIndexParamsMS, gd);
    cuvsCagraIndexParams.build_algo(cagraIndexParamsMS, ba.label);
    cuvsCagraIndexParams.nn_descent_niter(cagraIndexParamsMS, nndn);
  }

  public static class Builder {

    int igd = 128;
    int gd = 64;
    CuvsCagraGraphBuildAlgo ba = CuvsCagraGraphBuildAlgo.IVF_PQ;
    int nndn = 20;
    Arena arena;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withIntermediateGraphDegree(int igd) {
      this.igd = igd;
      return this;
    }

    public Builder withGraphDegree(int gd) {
      this.gd = gd;
      return this;
    }

    public Builder withBuildAlgo(CuvsCagraGraphBuildAlgo ba) {
      this.ba = ba;
      return this;
    }

    public Builder withNNDescentNiter(int nndn) {
      this.nndn = nndn;
      return this;
    }

    public CagraIndexParams build() throws Throwable {
      return new CagraIndexParams(arena, igd, gd, ba, nndn);
    }

  }

}
