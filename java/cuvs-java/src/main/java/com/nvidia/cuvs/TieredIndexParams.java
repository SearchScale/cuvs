/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
package com.nvidia.cuvs;

import java.util.Objects;
import java.util.Optional;

/**
 * Parameters for constructing a {@link TieredIndex}.
 * Only CAGRA is supported for now.
 */
public final class TieredIndexParams {

    public enum Metric {
        L2, INNER_PRODUCT
    }

    private final Metric metric;
    private final int minAnnRows;
    private final boolean createAnnIndexOnExtend;
    private final CagraIndexParams cagraParams;

    private TieredIndexParams(Builder builder) {
        this.metric = builder.metric;
        this.minAnnRows = builder.minAnnRows;
        this.createAnnIndexOnExtend = builder.createAnnIndexOnExtend;
        this.cagraParams = builder.cagraParams;
    }

    public Metric getMetric() {
        return metric;
    }

    public int getMinAnnRows() {
        return minAnnRows;
    }

    public boolean isCreateAnnIndexOnExtend() {
        return createAnnIndexOnExtend;
    }

    public CagraIndexParams getCagraParams() {
        return cagraParams;
    }

    public static Builder newBuilder() {
        return new Builder();
    }

    public static final class Builder {
        private Metric metric = Metric.L2;
        private int minAnnRows = 4096;
        private boolean createAnnIndexOnExtend = true;
        private CagraIndexParams cagraParams = null;

        public Builder metric(Metric metric) {
            this.metric = Objects.requireNonNull(metric);
            return this;
        }

        public Builder minAnnRows(int minAnnRows) {
            this.minAnnRows = minAnnRows;
            return this;
        }

        public Builder createAnnIndexOnExtend(boolean val) {
            this.createAnnIndexOnExtend = val;
            return this;
        }

        public Builder withCagraParams(CagraIndexParams params) {
            this.cagraParams = Objects.requireNonNull(params);
            return this;
        }

        public TieredIndexParams build() {
            if (cagraParams == null)
                throw new IllegalStateException("CAGRA params required");
            return new TieredIndexParams(this);
        }
    }
}
