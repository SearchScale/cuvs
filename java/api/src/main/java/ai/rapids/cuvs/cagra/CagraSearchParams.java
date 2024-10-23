package ai.rapids.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import ai.rapids.cuvs.panama.cuvsCagraSearchParams;

/*
* struct cuvsCagraSearchParams {
*     size_t max_queries;
*     size_t itopk_size;
*     size_t max_iterations;
*     enum cuvsCagraSearchAlgo algo;
*     size_t team_size;
*     size_t search_width;
*     size_t min_iterations;
*     size_t thread_block_size;
*     enum cuvsCagraHashMode hashmap_mode;
*     size_t hashmap_min_bitlen;
*     float hashmap_max_fill_rate;
*     uint32_t num_random_samplings;
*     uint64_t rand_xor_mask;
* }
*/
public class CagraSearchParams {

  Arena arena;
  int max_queries;
  int itopk_size;
  int max_iterations;
  CuvsCagraSearchAlgo algo;
  int team_size;
  int search_width;
  int min_iterations;
  int thread_block_size;
  CuvsCagraHashMode hashmap_mode;
  int hashmap_min_bitlen;
  float hashmap_max_fill_rate;
  int num_random_samplings;
  long rand_xor_mask;

  enum CuvsCagraSearchAlgo {
    SINGLE_CTA(0), MULTI_CTA(1), MULTI_KERNEL(2), AUTO(3);

    public final int label;

    private CuvsCagraSearchAlgo(int label) {
      this.label = label;
    }

  }

  enum CuvsCagraHashMode {
    HASH(0), SMALL(1), AUTO_HASH(2);

    public final int label;

    private CuvsCagraHashMode(int label) {
      this.label = label;
    }
  }

  public MemorySegment cagraSearchParamsMS;

  public CagraSearchParams(Arena arena, int max_queries, int itopk_size, int max_iterations, CuvsCagraSearchAlgo algo,
      int team_size, int search_width, int min_iterations, int thread_block_size, CuvsCagraHashMode hashmap_mode,
      int hashmap_min_bitlen, float hashmap_max_fill_rate, int num_random_samplings, long rand_xor_mask) {
    super();
    this.arena = arena;
    this.max_queries = max_queries;
    this.itopk_size = itopk_size;
    this.max_iterations = max_iterations;
    this.algo = algo;
    this.team_size = team_size;
    this.search_width = search_width;
    this.min_iterations = min_iterations;
    this.thread_block_size = thread_block_size;
    this.hashmap_mode = hashmap_mode;
    this.hashmap_min_bitlen = hashmap_min_bitlen;
    this.hashmap_max_fill_rate = hashmap_max_fill_rate;
    this.num_random_samplings = num_random_samplings;
    this.rand_xor_mask = rand_xor_mask;
    this.set();
  }

  public void set() {
    cagraSearchParamsMS = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(cagraSearchParamsMS, max_queries);
    cuvsCagraSearchParams.itopk_size(cagraSearchParamsMS, itopk_size);
    cuvsCagraSearchParams.max_iterations(cagraSearchParamsMS, max_iterations);
    cuvsCagraSearchParams.algo(cagraSearchParamsMS, algo.label);
    cuvsCagraSearchParams.team_size(cagraSearchParamsMS, team_size);
    cuvsCagraSearchParams.search_width(cagraSearchParamsMS, search_width);
    cuvsCagraSearchParams.min_iterations(cagraSearchParamsMS, min_iterations);
    cuvsCagraSearchParams.thread_block_size(cagraSearchParamsMS, thread_block_size);
    cuvsCagraSearchParams.hashmap_mode(cagraSearchParamsMS, hashmap_mode.label);
    cuvsCagraSearchParams.hashmap_min_bitlen(cagraSearchParamsMS, hashmap_min_bitlen);
    cuvsCagraSearchParams.hashmap_max_fill_rate(cagraSearchParamsMS, hashmap_max_fill_rate);
    cuvsCagraSearchParams.num_random_samplings(cagraSearchParamsMS, num_random_samplings);
    cuvsCagraSearchParams.rand_xor_mask(cagraSearchParamsMS, rand_xor_mask);
  }

  @Override
  public String toString() {
    return "CagraSearchParams [max_queries=" + max_queries + ", itopk_size=" + itopk_size + ", max_iterations="
        + max_iterations + ", algo=" + algo + ", team_size=" + team_size + ", search_width=" + search_width
        + ", min_iterations=" + min_iterations + ", thread_block_size=" + thread_block_size + ", hashmap_mode="
        + hashmap_mode + ", hashmap_min_bitlen=" + hashmap_min_bitlen + ", hashmap_max_fill_rate="
        + hashmap_max_fill_rate + ", num_random_samplings=" + num_random_samplings + ", rand_xor_mask=" + rand_xor_mask
        + "]";
  }

  public static class Builder {

    Arena arena;
    int max_queries = 1;
    int itopk_size = 2;
    int max_iterations = 3;
    CuvsCagraSearchAlgo algo = CuvsCagraSearchAlgo.MULTI_KERNEL;
    int team_size = 4;
    int search_width = 5;
    int min_iterations = 6;
    int thread_block_size = 7;
    CuvsCagraHashMode hashmap_mode = CuvsCagraHashMode.AUTO_HASH;
    int hashmap_min_bitlen = 8;
    float hashmap_max_fill_rate = 9.0f;
    int num_random_samplings = 10;
    long rand_xor_mask = 11L;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withMaxQueries(int max_queries) {
      this.max_queries = max_queries;
      return this;
    }

    public Builder withItopkSize(int itopk_size) {
      this.itopk_size = itopk_size;
      return this;
    }

    public Builder withMaxIterations(int max_iterations) {
      this.max_iterations = max_iterations;
      return this;
    }

    public Builder withAlgo(CuvsCagraSearchAlgo algo) {
      this.algo = algo;
      return this;
    }

    public Builder withTeamSize(int team_size) {
      this.team_size = team_size;
      return this;
    }

    public Builder withSearchWidth(int search_width) {
      this.search_width = search_width;
      return this;
    }

    public Builder withMinIterations(int min_iterations) {
      this.min_iterations = min_iterations;
      return this;
    }

    public Builder withThreadBlockSize(int thread_block_size) {
      this.thread_block_size = thread_block_size;
      return this;
    }

    public Builder withHashmapMode(CuvsCagraHashMode hashmap_mode) {
      this.hashmap_mode = hashmap_mode;
      return this;
    }

    public Builder withHashmapMinBitlen(int hashmap_min_bitlen) {
      this.hashmap_min_bitlen = hashmap_min_bitlen;
      return this;
    }

    public Builder withHashmapMaxFillRate(float hashmap_max_fill_rate) {
      this.hashmap_max_fill_rate = hashmap_max_fill_rate;
      return this;
    }

    public Builder withNumRandomSamplings(int num_random_samplings) {
      this.num_random_samplings = num_random_samplings;
      return this;
    }

    public Builder withRandXorMask(long rand_xor_mask) {
      this.rand_xor_mask = rand_xor_mask;
      return this;
    }

    public CagraSearchParams build() throws Throwable {
      return new CagraSearchParams(arena, max_queries, itopk_size, max_iterations, algo, team_size, search_width,
          min_iterations, thread_block_size, hashmap_mode, hashmap_min_bitlen, hashmap_max_fill_rate,
          num_random_samplings, rand_xor_mask);
    }

  }
}