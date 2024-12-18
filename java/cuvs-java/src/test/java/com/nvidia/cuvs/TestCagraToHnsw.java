package com.nvidia.cuvs;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileOutputStream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.common.Util;

public class TestCagraToHnsw {

    private static final Logger log = LoggerFactory.getLogger(TestCagraToHnsw.class);

    @Test
    public void testSerialization() throws Throwable {
        String cagraFilePath = "cagra_index.bin";
        String hnswFilePath = "hnsw_index.bin";

        try (CuVSResources resources = new CuVSResources()) {
            // Build and serialize a CAGRA index
            float[][] dataset = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f},
                {7.0f, 8.0f, 9.0f}
            };

            CagraIndex cagraIndex = new CagraIndex.Builder(resources)
                .withDataset(dataset)
                .withIndexParams(new CagraIndexParams.Builder(resources).build())
                .build();

            // Create a temporary file for intermediate serialization
            File tempFile = File.createTempFile("cagra_temp_", ".tmp");

            // Serialize CAGRA index to a file
            try (FileOutputStream outputStream = new FileOutputStream(cagraFilePath)) {
                cagraIndex.serialize(outputStream, tempFile);
            }

            // Convert the CAGRA index to HNSW format
            Util.serializeCagraToHnsw(resources, cagraFilePath, hnswFilePath);

            // Verify the HNSW index file
            File hnswFile = new File(hnswFilePath);
            assertTrue("HNSW index file should exist", hnswFile.exists());
            log.info("HNSW index file successfully created at: {}", hnswFilePath);
        }
    }
}
