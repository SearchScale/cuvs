package ai.rapids.cuvs.cagra;

import java.io.InputStream;

public class CagraIndex {
	
	private CagraIndexParams params;
	private final float[][] dataset;
	private final CuVSResources res;
	private CagraIndexReference ref;

	// Constructor that takes build params and dataset, creates an index
	private CagraIndex(CagraIndexParams params, float[][] dataset, CuVSResources res) {
		this.params = params;
		this.dataset = dataset;
		this.res = res;
		
		this.ref = build(); 
	}

	// Constructor that takes pre-built cagra index as bytes, deserializes them into an index
	private CagraIndex(byte[] bytes, CuVSResources res) {
		this.params = null;
		this.dataset = null;
		this.res = res;

		this.ref = deserialize(bytes); 
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
		
	public InputStream serialize() {
		return null;
	}
	
	private CagraIndexReference deserialize(byte[] bytes) {
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
		
		public Builder(CuVSResources res){
			this.res = res;
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
			return new CagraIndex(params, dataset, res);
		}
	}
	
}
