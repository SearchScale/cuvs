package com.nvidia.cuvs.common;

import com.nvidia.cuvs.CuVSResources;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class TestUtil {

  private static final Logger log = LoggerFactory.getLogger(TestUtil.class);

  @Test
  public void testGpuDetails() {
    try {
      CuVSResources resources = new CuVSResources();

      int maxGpus = 10;
      int maxDetailLength = 256;

      GpuDetail[] gpuDetails = Util.getGpuDetails(resources, maxGpus, maxDetailLength);

      assertNotNull("GPU details should not be null", gpuDetails);
      assertTrue("GPU details array should contain at least one GPU", gpuDetails.length > 0);

      log.info("Number of GPUs: {}", gpuDetails.length);
      for (GpuDetail detail : gpuDetails) {
        log.info("GPU Name: {}", detail.getName());
        log.info("Total Memory (MB): {}", detail.getTotalMemory());
        log.info("Free Memory (MB): {}", detail.getFreeMemory());
      }

    } catch (Throwable e) {
      log.error("Test failed due to an exception: {}", e.getMessage(), e);
      throw new RuntimeException("Test failed due to an exception: " + e.getMessage(), e);
    }
  }
}
