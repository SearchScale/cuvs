package com.nvidia.cuvs.cagra;

import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.VarHandle;
import java.util.HashMap;
import java.util.Map;

public class SearchResult {

    private Map<Integer, Map<Integer, Float>> results; // Stores results for multiple queries
    private Map<Integer, Integer> mapping;
    SequenceLayout neighboursSL;
    SequenceLayout distancesSL;
    MemorySegment neighboursMS;
    MemorySegment distancesMS;
    int topK;

    /**
     * Constructor for search results loaded from memory.
     */
    public SearchResult(SequenceLayout neighboursSL, SequenceLayout distancesSL, MemorySegment neighboursMS,
                        MemorySegment distancesMS, int topK, Map<Integer, Integer> mapping) {
        super();
        this.topK = topK;
        this.neighboursSL = neighboursSL;
        this.distancesSL = distancesSL;
        this.neighboursMS = neighboursMS;
        this.distancesMS = distancesMS;
        this.mapping = mapping;
        this.results = new HashMap<>();
        this.load();
    }

    /**
     * Constructor for brute force or precomputed results.
     */
    public SearchResult(Map<Integer, Map<Integer, Float>> results) {
        this.results = results;
    }

    /**
     * Load results from memory segments for multiple queries.
     */
    private void load() {
        VarHandle neighboursVH = neighboursSL.varHandle(PathElement.sequenceElement());
        VarHandle distancesVH = distancesSL.varHandle(PathElement.sequenceElement());

        for (long queryIndex = 0; queryIndex < topK; queryIndex++) {
            Map<Integer, Float> queryResults = new HashMap<>();
            for (long i = 0; i < topK; i++) {
                int id = (int) neighboursVH.get(neighboursMS, queryIndex, i);
                float distance = (float) distancesVH.get(distancesMS, queryIndex, i);
                queryResults.put(mapping != null ? mapping.get(id) : id, distance);
            }
            results.put((int) queryIndex, queryResults);
        }
    }

    /**
     * Retrieve results for a specific query.
     */
    public Map<Integer, Float> getResults(int queryIndex) {
        return results.get(queryIndex);
    }

    /**
     * Retrieve results for all queries.
     */
    public Map<Integer, Map<Integer, Float>> getAllResults() {
        return results;
    }
}
