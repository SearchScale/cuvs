package com.nvidia.cuvs.common;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.nvidia.cuvs.CuVSResources;

public class TestUtil {

  @Test
  public void testGpuDetails() throws Throwable {
    try {
      CuVSResources resources = new CuVSResources();
      String details = Util.getGpuDetails(resources, 10, 256);
      System.out.println("GPU Details: " + details);
      assertTrue("GPU details should not be empty", !details.isEmpty());
    } catch (RuntimeException e) {
      e.printStackTrace();
      throw new AssertionError("Test failed due to an exception: " + e.getMessage());
    }
  }

}
