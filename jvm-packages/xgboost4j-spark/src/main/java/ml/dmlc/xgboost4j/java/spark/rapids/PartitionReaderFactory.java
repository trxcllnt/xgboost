/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.xgboost4j.java.spark.rapids;

import java.io.Serializable;

import org.apache.spark.sql.execution.datasources.FilePartition;


/**
 * A factory used to create {@link PartitionReader} instances.
 *
 * If Spark fails to execute any methods in the implementations of this interface or in the returned
 * {@link PartitionReader} (by throwing an exception), corresponding Spark task would fail and
 * get retried until hitting the maximum retry times.
 */

public interface PartitionReaderFactory extends Serializable {

  /**
   * Returns a columnar partition reader to read data from the given {@link FilePartition}.
   *
   * Implementations probably need to cast the input partition to the concrete
   * {@link FilePartition} class defined for the data source.
   */
  default PartitionReader<GpuColumnBatch> createColumnarReader(FilePartition partition) {
    throw new UnsupportedOperationException("Cannot create columnar reader.");
  }

  /**
   * Returns true if the given {@link FilePartition} should be read by Spark in a columnar way.
   * This means, implementations must also implement { #createColumnarReader(FilePartition)}
   * for the input partitions that this method returns true.
   *
   * As of Spark 2.4, Spark can only read all input partition in a columnar way, or none of them.
   * Data source can't mix columnar and row-based partitions. This may be relaxed in future
   * versions.
   */
  default boolean supportColumnarReads(FilePartition partition) {
    return false;
  }
}
