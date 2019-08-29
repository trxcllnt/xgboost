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

import java.sql.{Date, Timestamp}

import ai.rapids.cudf.Cuda
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.SparkConf
import org.scalatest.FunSuite

class OrcScanSuite extends FunSuite with PerTest with SparkQueryCompareTestSuite{
  private lazy val SAMPLE_ORC_PATH = "/sample.orc"

  test("Test Orc parsing should be same for cpu and gpu ") {
    assume(Cuda.isEnvCompatibleForTesting)

    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.orc(getTestDataPath("/file-splits.orc")),
      new GpuDataReader(ss)
        .option("asFloats", false)
        .orc(getTestDataPath("/file-splits.orc")),
      conf = new SparkConf(),
      repart = 1)
    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("Test Orc Chunk") {
    assume(Cuda.isEnvCompatibleForTesting)
    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.orc(getTestDataPath("/file-splits.orc")),
      new GpuDataReader(ss)
        .option("maxRowsPerChunk", 2048)
        .option("asFloats", false)
        .orc(getTestDataPath("/file-splits.orc")),
      conf = new SparkConf(),
      repart = 1)

    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("ORC parsing with raw types") {
    assume(Cuda.isEnvCompatibleForTesting)
    val data = new GpuDataReader(ss)
      .option("asFloats", false)
      .orc(getTestDataPath(SAMPLE_ORC_PATH))
      .buildRDD
      .mapPartitions(GpuDataset.columnBatchToRows)
      .collect
    assertResult(2) { data.length }
    val firstRow = data.head
    val secondRow = data.last
    assertResult(false) { firstRow.getBoolean(0) }
    assertResult(2) {  firstRow.getInt(1) }
    assertResult(6.25) { firstRow.getDouble(3) }
    assertResult(new Timestamp(1564999726000L)) { firstRow.getTimestamp(5) }
    assertResult(true) { secondRow.getBoolean(0) }
    assertResult(5.5) { secondRow.getFloat(2) }
    assertResult(Date.valueOf("2019-08-06")) { secondRow.getDate(4) }
  }

  test("ORC split when loading, stripes not divided") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 2048)
    assume(Cuda.isEnvCompatibleForTesting)
    val dataPath = getTestDataPath("/2000rows.orc")
    val dataset = new GpuDataReader(ss)
      .orc(dataPath)
    assertResult(2) {dataset.partitions.length}
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    assertResult(2) {rdd.getNumPartitions}
    val counts = rdd.collect
    orcUnproperSplitVerification(counts)

  }


  test("ORC split when loading, stripes get divided") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 1024)
    assume(Cuda.isEnvCompatibleForTesting)
    val dataPath = getTestDataPath("/2000rows.orc")
    val dataset = new GpuDataReader(ss)
      .orc(dataPath)
    assertResult(4) {dataset.partitions.length}
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    assertResult(4) {rdd.getNumPartitions}
    val counts = rdd.collect
    orcProperSplitVerification(counts)
  }

  test("ORC repartition, stripes get divided") {
    assume(Cuda.isEnvCompatibleForTesting)
    val dataPath = getTestDataPath("/2000rows.orc")
    val dataset = new GpuDataReader(ss)
      .orc(dataPath)
    val newDataset = dataset.repartition(4)
    assertResult(4) {newDataset.partitions.length}
    val rdd = newDataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)

    assertResult(4) {rdd.getNumPartitions}
    val counts = rdd.collect
    assertResult(8) { counts.length }
    orcProperSplitVerification(counts)
  }

  test("ORC repartition, stripes not divided") {
    assume(Cuda.isEnvCompatibleForTesting)
    val dataPath = getTestDataPath("/2000rows.orc")
    val dataset = new GpuDataReader(ss)
      .orc(dataPath)
    val newDataset = dataset.repartition(2)
    assertResult(2) {newDataset.partitions.length}
    val rdd = newDataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    assertResult(2) {rdd.getNumPartitions}
    val counts = rdd.collect
    orcUnproperSplitVerification(counts)
  }

  private def orcProperSplitVerification(counts: Array[(Long)]): Unit = {
    assertResult(8) {counts.length}
    assertResult(6) { counts(0) }
    assertResult(1024) { counts(1) }
    assertResult(6){ counts(2) }
    assertResult(976) { counts(3) }
    assertResult(0) { counts(4) }
    assertResult(0) { counts(5) }
    assertResult(0) { counts(6) }
    assertResult(0) { counts(7) }
  }

  private def orcUnproperSplitVerification(counts: Array[(Long)]): Unit = {
    assertResult(4) { counts.length }
    assertResult(6) {counts(0)}
    assertResult(2000) {counts(1)}
    assertResult(0) {counts(2)}
    assertResult(0) {counts(3)}
  }

  test("Test Orc rows chunk") {
    assume(Cuda.isEnvCompatibleForTesting)
    val fileName = getTestDataPath("/file-splits.orc")

    val reader = new GpuDataReader(ss)

    var dataset = reader.option("maxRowsPerChunk", 20).orc(fileName, fileName)
      .repartition(1)
    var counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(10) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 1026).orc(fileName, fileName)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(10) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 2049).orc(fileName, fileName)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(6) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 3097).orc(fileName, fileName)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(4) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 4096).orc(fileName, fileName)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(4) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 10000).orc(fileName, fileName)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(2) { counts(0) }

  }
}
