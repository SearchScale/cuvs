package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;

import com.nvidia.cuvs.NativeFloatBufferProvider;

public class NativeFloatBufferProviderImpl extends NativeFloatBufferProvider {

	final Arena arena;
	final MemorySegment seg;
	
	public NativeFloatBufferProviderImpl(long size) {
		this.arena = Arena.ofConfined();
		this.seg = arena.allocate(MemoryLayout.sequenceLayout(size, C_FLOAT));
	}
	@Override
	public void close() throws Exception {
		arena.close();
	}

	@Override
	public FloatBuffer getBuffer() {
		// TODO Auto-generated method stub
		return seg.asByteBuffer().asFloatBuffer();
	}

}
