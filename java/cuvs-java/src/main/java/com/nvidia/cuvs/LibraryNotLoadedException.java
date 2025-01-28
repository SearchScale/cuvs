package com.nvidia.cuvs;

public class LibraryNotLoadedException extends RuntimeException {

  private static final long serialVersionUID = -3343802355148302803L;

  public LibraryNotLoadedException(Exception ex) {
    super(ex);
  }

  public LibraryNotLoadedException(String message) {
    super(message);
  }
}
