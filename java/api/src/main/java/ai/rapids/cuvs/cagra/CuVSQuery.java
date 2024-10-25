package ai.rapids.cuvs.cagra;

import java.util.Arrays;

public class CuVSQuery {

  CagraSearchParams searchParams;
  PreFilter preFilter;
  float[][] queryVectors;

  public CuVSQuery(CagraSearchParams searchParams, PreFilter preFilter, float[][] queryVectors) {
    super();
    this.searchParams = searchParams;
    this.preFilter = preFilter;
    this.queryVectors = queryVectors;
  }

  @Override
  public String toString() {
    return "CuVSQuery [searchParams=" + searchParams + ", preFilter=" + preFilter + ", queries="
        + Arrays.toString(queryVectors) + "]";
  }

  public CagraSearchParams getSearchParams() {
    return searchParams;
  }

  public PreFilter getPreFilter() {
    return preFilter;
  }

  public float[][] getQueries() {
    return queryVectors;
  }

  public static class Builder {
    CagraSearchParams searchParams;
    PreFilter preFilter;
    float[][] queryVectors;

    /**
     * 
     * @param res
     */
    public Builder() {
    }

    /**
     * 
     * @param dataset
     * @return
     */
    public Builder withSearchParams(CagraSearchParams searchParams) {
      this.searchParams = searchParams;
      return this;
    }

    /**
     * 
     * @param queryVectors
     * @return
     */
    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * 
     * @param preFilter
     * @return
     */
    public Builder withPreFilter(PreFilter preFilter) {
      this.preFilter = preFilter;
      return this;
    }

    /**
     * 
     * @return
     * @throws Throwable
     */
    public CuVSQuery build() throws Throwable {
      return new CuVSQuery(searchParams, preFilter, queryVectors);
    }
  }

}
