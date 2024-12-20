/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.cuvs.common;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.IOUtils;

import com.nvidia.cuvs.CuVSResources;

public class Util {
  
  public static MemorySegment toCString(Arena arena, String string) {
    byte[] bytes = (string + "\0").getBytes();
    MemorySegment segment = arena.allocate(bytes.length);
    segment.copyFrom(MemorySegment.ofArray(bytes));
    return segment;
  }

  /**
   * A utility method for getting an instance of {@link MemorySegment} for a
   * {@link String}.
   * 
   * @param str the string for the expected {@link MemorySegment}
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, String str) {
    StringBuilder sb = new StringBuilder(str).append('\0');
    MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(sb.length(), linker.canonicalLayouts().get("char"));
    MemorySegment stringMemorySegment = arena.allocate(stringMemoryLayout);

    for (int i = 0; i < sb.length(); i++) {
      VarHandle varHandle = stringMemoryLayout.varHandle(PathElement.sequenceElement(i));
      varHandle.set(stringMemorySegment, 0L, (byte) sb.charAt(i));
    }
    return stringMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 2D float array.
   * 
   * @param data The 2D float array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, float[][] data) {
    long rows = data.length;
    long cols = rows > 0 ? data[0].length : 0;

    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows * cols, linker.canonicalLayouts().get("float"));
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    long floatByteSize = linker.canonicalLayouts().get("float").byteSize();

    for (int r = 0; r < rows; r++) {
      MemorySegment.copy(data[r], 0, dataMemorySegment, (ValueLayout) linker.canonicalLayouts().get("float"),
          (r * cols * floatByteSize), (int) cols);
    }

    return dataMemorySegment;
  }

  /**
   * Load a file from the classpath to a temporary file. Suitable for loading .so
   * files from the jar.
   */
  public static File loadLibraryFromJar(String path) throws IOException {
    if (!path.startsWith("/")) {
      throw new IllegalArgumentException("The path has to be absolute (start with '/').");
    }
    // Obtain filename from path
    String[] parts = path.split("/");
    String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

    // Split filename to prefix and suffix (extension)
    String prefix = "";
    String suffix = null;
    if (filename != null) {
      parts = filename.split("\\.", 2);
      prefix = parts[0];
      suffix = (parts.length > 1) ? "." + parts[parts.length - 1] : null;
    }
    // Prepare temporary file
    File temp = File.createTempFile(prefix, suffix);
    IOUtils.copy(Util.class.getResourceAsStream(path), new FileOutputStream(temp));

    return temp;
  }

  public static void convertCagraToHnsw(CuVSResources resources, String cagraFilePath, String hnswFilePath) {
    try {
      MethodHandle convertCagraToHnswHandle = resources.cagraToHnswHandle;

      byte[] cagraBytes = cagraFilePath.getBytes(StandardCharsets.UTF_8);
      byte[] hnswBytes = hnswFilePath.getBytes(StandardCharsets.UTF_8);

      MemorySegment cagraFileSegment = resources.arena.allocate(cagraBytes.length + 1);
      MemorySegment hnswFileSegment = resources.arena.allocate(hnswBytes.length + 1);

      cagraFileSegment.asSlice(0, cagraBytes.length).copyFrom(MemorySegment.ofArray(cagraBytes));
      cagraFileSegment.set(ValueLayout.JAVA_BYTE, cagraBytes.length, (byte) 0);

      hnswFileSegment.asSlice(0, hnswBytes.length).copyFrom(MemorySegment.ofArray(hnswBytes));
      hnswFileSegment.set(ValueLayout.JAVA_BYTE, hnswBytes.length, (byte) 0);

      int returnValue = (int) convertCagraToHnswHandle.invoke(resources.getMemorySegment(), cagraFileSegment,
          hnswFileSegment);

      if (returnValue != 0) {
        throw new RuntimeException("Failed to convert CAGRA to HNSW, error code: " + returnValue);
      }
    } catch (Throwable e) {
      throw new RuntimeException("Error during CAGRA to HNSW conversion", e);
    }
  }

}
