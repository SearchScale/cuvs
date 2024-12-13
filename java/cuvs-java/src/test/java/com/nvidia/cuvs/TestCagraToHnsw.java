package com.nvidia.cuvs;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.nvidia.cuvs.common.Util;

public class TestCagraToHnsw {

    private static final Logger log = LoggerFactory.getLogger(TestCagraToHnsw.class);

    @Test
    public void testCagraToHnswConversion() throws Throwable {
        String cagraFilePath = "cagra_index.bin";
        String hnswFilePath = "hnsw_index.bin";

        try (CuVSResources resources = new CuVSResources()) {
            float[][] dataset = { 
                { 1.0f, 2.0f, 3.0f }, 
                { 4.0f, 5.0f, 6.0f }, 
                { 7.0f, 8.0f, 9.0f } 
            };

            // Build CAGRA index
            CagraIndexParams indexParams = new CagraIndexParams.Builder(resources).build();
            CagraIndex index = new CagraIndex.Builder(resources)
                .withDataset(dataset)
                .withIndexParams(indexParams)
                .build();

            // Serialize CAGRA index to file using OutputStream
            try (OutputStream outputStream = new FileOutputStream(cagraFilePath)) {
                index.serialize(outputStream, new File(cagraFilePath));
                log.info("CAGRA index successfully created at: {}", cagraFilePath);
            }

            Util.convertCagraToHnsw(resources, cagraFilePath, hnswFilePath);
            log.info("HNSW index conversion completed. File created at: {}", hnswFilePath);

            File hnswFile = new File(hnswFilePath);
            assertTrue("HNSW index file should exist", hnswFile.exists());

            Path hnswPath = Path.of(hnswFilePath);
            long hnswFileSize = Files.size(hnswPath);
            assertTrue("HNSW index file should not be empty", hnswFileSize > 0);
        }
    }
}
