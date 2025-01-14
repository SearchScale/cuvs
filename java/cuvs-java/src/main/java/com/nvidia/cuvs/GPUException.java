package com.nvidia.cuvs;

public class GPUException extends RuntimeException {

  private static final long serialVersionUID = 3634160877601102571L;

  public GPUException(Exception ex) {
    super(ex);
  }

  public GPUException(String message) {
    super(message);
  }
}
