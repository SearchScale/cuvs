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

  Arena arena;
  int intermediate_graph_degree;
  int graph_degree;
  CuvsCagraGraphBuildAlgo build_algo;
  int nn_descent_niter;

  public enum CuvsCagraGraphBuildAlgo {
    AUTO_SELECT(0), IVF_PQ(1), NN_DESCENT(2);

    public final int label;

    private CuvsCagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  public MemorySegment cagraIndexParamsMS;

  public CagraIndexParams(Arena arena, int intermediate_graph_degree, int graph_degree,
      CuvsCagraGraphBuildAlgo build_algo, int nn_descent_niter) {
    this.arena = arena;
    this.intermediate_graph_degree = intermediate_graph_degree;
    this.graph_degree = graph_degree;
    this.build_algo = build_algo;
    this.nn_descent_niter = nn_descent_niter;
    this.set();
  }

  private void set() {
    cagraIndexParamsMS = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(cagraIndexParamsMS, intermediate_graph_degree);
    cuvsCagraIndexParams.graph_degree(cagraIndexParamsMS, graph_degree);
    cuvsCagraIndexParams.build_algo(cagraIndexParamsMS, build_algo.label);
    cuvsCagraIndexParams.nn_descent_niter(cagraIndexParamsMS, nn_descent_niter);
  }

  @Override
  public String toString() {
    return "CagraIndexParams [intermediate_graph_degree=" + intermediate_graph_degree + ", graph_degree=" + graph_degree
        + ", build_algo=" + build_algo + ", nn_descent_niter=" + nn_descent_niter + "]";
  }

  public static class Builder {

    Arena arena;
    int intermediate_graph_degree = 128;
    int graph_degree = 64;
    CuvsCagraGraphBuildAlgo build_algo = CuvsCagraGraphBuildAlgo.IVF_PQ;
    int nn_descent_niter = 20;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withIntermediateGraphDegree(int intermediate_graph_degree) {
      this.intermediate_graph_degree = intermediate_graph_degree;
      return this;
    }

    public Builder withGraphDegree(int graph_degree) {
      this.graph_degree = graph_degree;
      return this;
    }

    public Builder withBuildAlgo(CuvsCagraGraphBuildAlgo build_algo) {
      this.build_algo = build_algo;
      return this;
    }

    public Builder withNNDescentNiter(int nn_descent_niter) {
      this.nn_descent_niter = nn_descent_niter;
      return this;
    }

    public CagraIndexParams build() throws Throwable {
      return new CagraIndexParams(arena, intermediate_graph_degree, graph_degree, build_algo, nn_descent_niter);
    }

  }

}
