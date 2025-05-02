package com.nvidia.cuvs;

import java.nio.FloatBuffer;

public abstract class NativeFloatBufferProvider implements AutoCloseable {
	public abstract FloatBuffer getBuffer();
}
