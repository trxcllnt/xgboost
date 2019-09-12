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
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.scalatest.FunSuite

class CsvScanSuite extends FunSuite with PerTest with SparkQueryCompareTestSuite {
  private lazy val TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val TEST_CSV_PATH = getTestDataPath("/rank.test.csv")

  test("Test CSV parsing should be same for cpu and gpu ") {
    assume(Cuda.isEnvCompatibleForTesting)
    val testAppPath = getTestDataPath("/test.csv")

    val schema = StructType(Array(
      StructField("a", IntegerType),
      StructField("b", IntegerType),
      StructField("c", IntegerType),
      StructField("d", IntegerType),
      StructField("e", IntegerType)))

    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.schema(schema).csv(testAppPath),
      new GpuDataReader(ss)
        .option("asFloats", false)
        .schema(schema)
        .csv(testAppPath),
      conf = new SparkConf(),
      repart = 1)
    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("CSV rows chunk") {
    assume(Cuda.isEnvCompatibleForTesting)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val reader = new GpuDataReader(ss).schema(csvSchema)

    var dataset = reader.option("maxRowsPerChunk", 100).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
      .repartition(1)
    var counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(3) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 1).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(215) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 10).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(22) { counts(0) }

    dataset = reader.option("maxRowsPerChunk", 1000).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
      .repartition(1)
    counts = getChunkCount(dataset)
    assertResult(1) { counts.length }
    assertResult(2) { counts(0) }
  }

  test("CSV rows columns number") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader
      .option("maxRowsPerChunk", 100)
      .schema(csvSchema).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
      .repartition(1)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)_1 }
    assertResult(215) { counts(0)_2 }
  }

  test("CSV parsing with compression") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader
      .option("maxRowsPerChunk", 100)
      .schema(csvSchema)
      .csv(getTestDataPath("/rank.train.csv.gz"))
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)_1 }
    assertResult(149) { counts(0)_2 }
  }

  test("Unknown CSV parsing option") {
    val reader = new GpuDataReader(ss)
    assertThrows[UnsupportedOperationException] {
      reader.option("something", "something").csv(TRAIN_CSV_PATH)
        .mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    }
  }

  test("CSV parsing with header") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/header.csv")
    val dataset = reader
      .option("maxRowsPerChunk", 5)
      .schema(csvSchema).option("header", true).csv(path)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)_1 }
    assertResult(10) { counts(0)_2 }
  }

  test("CSV parsing with custom delimiter") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/custom-delim.csv")
    val dataset = reader
      .option("maxRowsPerChunk", 5)
      .schema(csvSchema)
      .option("sep", "|").csv(path)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)_1 }
    assertResult(10) { counts(0)_2 }
  }

  test("CSV parsing with comments") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/commented.csv")
    val dataset = reader
      .option("maxRowsPerChunk", 5)
      .schema(csvSchema)
      .option("comment", "#").csv(path)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)_1 }
    assertResult(10) { counts(0)_2 }
  }

  test("CSV multifile partition") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d INT"
    val path = getTestDataPath("/multifile-norank-train-csv")
    val dataset = reader
      .schema(csvSchema)
      .option("maxRowsPerChunk", 25)
      .csv(path)
      .repartition(1)
    val rdd = dataset.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper)
    assertResult(1) {
      rdd.partitions.length
    }
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(4) { counts(0)_1 }
    assertResult(149) { counts(0)_2 }
  }

  test(testName = "auto split csv file when loading") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 30)
    assume(Cuda.isEnvCompatibleForTesting)

    val reader = new GpuDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE"
    val dataset = reader.schema(csvSchema).csv(getTestDataPath("/test.csvsplit.csv"))
    val firstPartFile = dataset.partitions(0).files(0)
    val secondPartFile = dataset.partitions(1).files(0)
    assertResult(2) { dataset.partitions.length }
    assertResult(50) {dataset.partitions.flatMap(_.files).map(_.length).sum}
    assertResult(30) {firstPartFile.length}
    assertResult(30) {secondPartFile.start}
    assertResult(20) {secondPartFile.length}

    val rdd = dataset.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // for only one batch test
      if (iter.hasNext) {
        val b = iter.next()
        Iterator.single(b.getColumnVector(0).sum().getLong,
          b.getColumnVector(1).sum().getLong,
          b.getColumnVector(2).sum().getLong,
          b.getColumnVector(3).sum().getLong,
          b.getColumnVector(4).sum().getLong)
      } else {
        Iterator.empty
      }})
    val counts = rdd.collect
    csvSplitColumnSumVerification(counts)
  }

  test(testName = "csv repartition for numPartitions is greater than numPartitionedFiles") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE"
    val dataset = reader.schema(csvSchema).csv(getTestDataPath("/test.csvsplit.csv"))
    val newDataset = dataset.repartition(2)
    assertResult(2) {newDataset.partitions.length}
    assertResult(1) {newDataset.partitions(0).files.length}
    assertResult(true) {
      newDataset.partitions(1).files(0).start == newDataset.partitions(0).files(0).length}

    val rdd = newDataset.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // for only one batch test
      if (iter.hasNext) {
        val b = iter.next()
        Iterator.single(b.getColumnVector(0).sum().getLong,
          b.getColumnVector(1).sum().getLong,
          b.getColumnVector(2).sum().getLong,
          b.getColumnVector(3).sum().getLong,
          b.getColumnVector(4).sum().getLong)
      } else {
        Iterator.empty
      }})
    assertResult(2) {rdd.getNumPartitions}
    val counts = rdd.collect
    csvSplitColumnSumVerification(counts)
  }

  test("csv split when numPartitions is greater than numRows") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 3)
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE"
    val dataset = reader.schema(csvSchema).csv(getTestDataPath("/1line.csv"))

    val rdd = dataset.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // for only one batch test
      if (iter.hasNext) {
        val b = iter.next()
        Iterator.single(b.getColumnVector(0).sum().getLong,
          b.getColumnVector(1).sum().getLong,
          b.getColumnVector(2).sum().getLong)
      } else {
        Iterator.empty
      }})

    assertResult(2) {rdd.getNumPartitions}
    val counts = rdd.collect
    assertResult(1) {counts.length}
    assertResult(1) {counts(0)_1}
    assertResult(2) {counts(0)_2}
    assertResult(3) {counts(0)_3}
  }

  test("csv repartition when numPartitions is greater than total bytes") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE"
    val dataset = reader.schema(csvSchema).csv(getTestDataPath("/1line.csv"))
    val newDataset = dataset.repartition(10)
    assertResult(5) {newDataset.partitions.length}

    val rdd = newDataset.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // for only one batch test
      if (iter.hasNext) {
        val b = iter.next()
        Iterator.single(b.getColumnVector(0).sum().getLong,
          b.getColumnVector(1).sum().getLong,
          b.getColumnVector(2).sum().getLong)
      } else {
        Iterator.empty
      }
    })

    assertResult(5) {rdd.getNumPartitions}

    val counts = rdd.collect
    assertResult(1) {counts.length}
    assertResult(1) {counts(0)_1}
    assertResult(2) {counts(0)_2}
    assertResult(3) {counts(0)_3}
  }

  private def csvSplitColumnSumVerification(counts: Array[(Long, Long, Long, Long, Long)]):
      Unit = {
    assertResult(2) {counts.length}
    assertResult(1 + 6 + 11) {counts(0)._1}
    assertResult(2 + 7 + 12) {counts(0)._2}
    assertResult(3 + 8 + 13) {counts(0)._3}
    assertResult(4 + 9 + 14 ) {counts(0)._4}
    assertResult(5 + 10 + 15 ) {counts(0)._5}
    assertResult(16) {counts(1)._1}
    assertResult(17) {counts(1)._2}
    assertResult(18) {counts(1)._3}
    assertResult(19) {counts(1)._4}
    assertResult(20) {counts(1)._5}
  }
}
