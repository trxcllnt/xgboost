/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.scala.spark.rapids

import ai.rapids.cudf.Cuda
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.SparkConf
import org.scalatest.FunSuite

class ParquetScanSuite extends FunSuite with PerTest with SparkQueryCompareTestSuite {
  private lazy val TRAIN_PARQUET_PATH = getTestDataPath("/rank.train.parquet")

  test("Test Parquet load should be same for cpu and gpu ") {
    assume(Cuda.isEnvCompatibleForTesting)

    val fileName = "/rank.train.parquet"
    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.parquet(getTestDataPath(fileName)),
      new GpuDataReader(ss).option("asFloats", false).parquet(getTestDataPath(fileName)),
      conf = new SparkConf(),
      repart = 1)

    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("Test Parquet with chunks") {
    assume(Cuda.isEnvCompatibleForTesting)

    val fileName = "/file-splits.parquet"
    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.parquet(getTestDataPath(fileName)),
      new GpuDataReader(ss)
        .option("asFloats", false)
        .option("maxRowsPerChunk", 100)
        .parquet(getTestDataPath(fileName)),
      conf = new SparkConf(),
      repart = 1)

    assert(fromCpu.length == fromGpu.length)
    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("Parquet parsing") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val dataset = reader.parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(2) { counts.length }
    assertResult(5) { counts(0) }
    assertResult(149) { counts(1) }
  }

  test("Parquet subset parsing") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val specSchema = "a BOOLEAN, c DOUBLE, e INT"
    val dataset = reader.schema(specSchema).parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(2) { counts.length }
    assertResult(3) { counts(0) }
    assertResult(149) { counts(1) }
  }

  test("Parquet repartition when numPartitions is greater than numPartitionedFile") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val dataset = reader.parquet(TRAIN_PARQUET_PATH)
    assertResult(1) {dataset.partitions.length}

    val newDataset = dataset.repartition(2)
    assertResult(2) {newDataset.partitions.length}

    val rdd = newDataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)

    assertResult(2) {rdd.getNumPartitions}
    val counts = rdd.collect
    assertResult(4) {counts.length}
    assertResult(5) {counts(0) }
    assertResult(149) { counts(1) }
  }

  test("auto split when loading parquet file") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 3000)

    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val dataset = reader.parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)

    assertResult(2) { rdd.getNumPartitions }
    val counts = rdd.collect
    assertResult(4) { counts.length }
    assertResult(5) { counts(0) }
    assertResult(149) { counts(1) }
  }
}
