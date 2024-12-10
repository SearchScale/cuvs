package com.nvidia.cuvs.panama;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

import com.nvidia.cuvs.common.GpuDetail;

public class GpuDetailLayout {
  // Define the struct layout
  public static final MemoryLayout LAYOUT = MemoryLayout.structLayout(
      MemoryLayout.sequenceLayout(64, ValueLayout.JAVA_BYTE).withName("name"),
      ValueLayout.JAVA_LONG.withName("totalMemory"), ValueLayout.JAVA_LONG.withName("freeMemory"));

  public static final int MAX_NAME_LENGTH = 64;

  public static GpuDetail fromMemorySegment(MemorySegment segment) {
    // Extract fields from the memory segment
    String name = new String(segment.asSlice(0, MAX_NAME_LENGTH).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8)
        .trim();
    long totalMemory = segment.get(ValueLayout.JAVA_LONG, MAX_NAME_LENGTH);
    long freeMemory = segment.get(ValueLayout.JAVA_LONG, MAX_NAME_LENGTH + ValueLayout.JAVA_LONG.byteSize());

    return new GpuDetail(name, totalMemory, freeMemory);
  }
}
