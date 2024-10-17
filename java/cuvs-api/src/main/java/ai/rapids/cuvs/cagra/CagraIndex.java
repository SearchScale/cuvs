package ai.rapids.cuvs.cagra;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

public class CagraIndex {
	
	private CagraIndexParams params;
	private final float[][] dataset;
	private final CuVSResources res;
	private CagraIndexReference ref;
	
	private Map<Integer, Integer> mapping; // nocommit (this should be int[], not a mapping)

	// Constructor that takes build params and dataset, creates an index
	private CagraIndex(CagraIndexParams params, float[][] dataset, CuVSResources res) {
		this.params = params;
		this.dataset = dataset;
		this.res = res;
		
		this.ref = build(); 
	}

	// Constructor that takes pre-built cagra index as bytes, deserializes them into an index
	private CagraIndex(InputStream in, CuVSResources res) {
		this.params = null;
		this.dataset = null;
		this.res = res;

		this.ref = deserialize(in); 
	}

	// actual build method
	private CagraIndexReference build() {
		return null;
	}
	
	// search method
	public SearchResult search(CagraSearchParams params, float[][] queries) {
		// use ref to search, return search results
		return null;
	}

	public PointerToDataset getDataset() {
		
	}
	
	public void serialize(OutputStream out) {
		
	}
	
	private CagraIndexReference deserialize(InputStream in) {
		return null;
	}

	public CagraIndexParams getParams() {
		return params;
	}

	public float[][] getDataset() {
		return dataset;
	}

	public CuVSResources getResources() {
		return res;
	}

	public static class Builder {
		private CagraIndexParams params;
		float[][] dataset;
		CuVSResources res;
		
		InputStream in;
		
		public Builder(CuVSResources res){
			this.res = res;
		}
		
		public Builder from(InputStream in) {
			this.in = in;
			return this;
		}
		
		public Builder withDataset(float[][] dataset) {
			this.dataset = dataset;
			return this;
		}
		
		public Builder withIndexParams(CagraIndexParams params) {
			this.params = params;
			return this;
		}
		
		public CagraIndex build(){
			if (in != null) {
				return new CagraIndex(in, res);
			} else {
				return new CagraIndex(params, dataset, res);
			}
		}
	}
	
}
