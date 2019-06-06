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

package ml.dmlc.xgboost4j.scala.spark.nvidia

import ai.rapids.cudf.Cuda
import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class NVDatasetSuite extends FunSuite with PerTest {
  private lazy val TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val TRAIN_PARQUET_PATH = getTestDataPath("/rank.train.parquet")

  override def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
      .master("local[1]")
      .appName("NVDatasetSuite")
      .config("spark.ui.enabled", false)
      .config("spark.driver.memory", "512m")
      .config("spark.task.cpus", 1)

  test("mapColumnarSingleBatchPerPartition") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(TRAIN_CSV_PATH)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single(b.getNumColumns, b.getNumRows))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)._1 }
    assertResult(149) { counts(0)._2 }
  }

  test("Unknown CSV parsing option") {
    val reader = new NVDataReader(ss)
    assertThrows[UnsupportedOperationException] {
      reader.option("something", "something").csv(TRAIN_CSV_PATH)
        .mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
          Iterator.single(b.getNumColumns))
    }
  }

  test("CSV parsing with header") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/header.csv")
    val dataset = reader.schema(csvSchema).option("header", true).csv(path)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)._1 }
    assertResult(10) { counts(0)._2 }
  }

  test("CSV parsing with custom delimiter") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/custom-delim.csv")
    val dataset = reader.schema(csvSchema).option("sep", "|").csv(path)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)._1 }
    assertResult(10) { counts(0)._2 }
  }

  test("CSV parsing with comments") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val path = getTestDataPath("/commented.csv")
    val dataset = reader.schema(csvSchema).option("comment", "#").csv(path)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)._1 }
    assertResult(10) { counts(0)._2 }
  }

  test("CSV multifile partition") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d INT"
    val path = getTestDataPath("/multifile-norank-train-csv")
    val dataset = reader.schema(csvSchema).csv(path)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    assertResult(1) { rdd.partitions.length }
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(4) { counts(0)._1 }
    assertResult(149) { counts(0)._2 }
  }

  test("Parquet parsing") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val dataset = reader.parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0)._1 }
    assertResult(149) { counts(0)._2 }
  }

  test("Parquet subset parsing") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val specSchema = "a BOOLEAN, c DOUBLE, e INT"
    val dataset = reader.schema(specSchema).parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(3) { counts(0)._1 }
    assertResult(149) { counts(0)._2 }
  }

  test("buildRDD") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(ss)
    val dataPath = getTestDataPath("/rank.train.csv")
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(dataPath)
    val rdd = dataset.buildRDD.mapPartitions(_.flatMap(NVDataset.columnBatchToRows))
    val data = rdd.collect

    assertResult(149) { data.length }
    val firstRow = data.head
    assertResult(5) { firstRow.size }
    assertResult(false) { firstRow.getBoolean(0) }
    implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.000000001)
    assert(firstRow.getDouble(1) === 985.574005058)
    assert(firstRow.getDouble(2) === 320.223538037)
    assert(firstRow.getDouble(3) === 0.621236086198)
    assertResult(1) { firstRow.get(4) }

    val lastRow = data.last
    assertResult(5) { lastRow.size }
    assertResult(false) { firstRow.getBoolean(0) }
    assert(lastRow.getDouble(1) === 9.90744703306)
    assert(lastRow.getDouble(2) === 50.810461183)
    assert(lastRow.getDouble(3) === 3.00863325197)
    assertResult(20) { lastRow.get(4) }
  }

  test("repartition to 1") {
    val partFiles = Array(
      PartitionedFile(null, "/a", 0, 123),
      PartitionedFile(null, "/b", 0, 456),
      PartitionedFile(null, "/c", 0, 789),
      PartitionedFile(null, "/d", 0, 2468),
      PartitionedFile(null, "/e", 0, 3579),
      PartitionedFile(null, "/f", 0, 12345),
      PartitionedFile(null, "/g", 0, 67890)
    )
    val oldPartitions = Seq(
      FilePartition(0, Seq(partFiles(0))),
      FilePartition(1, partFiles.slice(1, 4)),
      FilePartition(2, partFiles.slice(4, 6)),
      FilePartition(3, Seq(partFiles(6)))
    )
    val oldset = new NVDataset(null, null, null, Some(oldPartitions))
    val newset = oldset.repartition(1)
    assertResult(1)(newset.partitions.length)
    assertResult(0)(newset.partitions(0).index)
    val newPartFiles = newset.partitions(0).files
    assertResult(7)(newPartFiles.length)
    for (f <- partFiles) {
      assert(newPartFiles.contains(f))
    }
  }

  test("repartition") {
    val partFiles = Array(
      PartitionedFile(null, "/a", 0, 1230),
      PartitionedFile(null, "/b", 0, 4560),
      PartitionedFile(null, "/c", 0, 7890),
      PartitionedFile(null, "/d", 0, 2468),
      PartitionedFile(null, "/e", 0, 3579),
      PartitionedFile(null, "/f", 0, 12345),
      PartitionedFile(null, "/g", 0, 67890)
    )
    val oldPartitions = Seq(
      FilePartition(0, Seq(partFiles(0))),
      FilePartition(1, partFiles.slice(1, 4)),
      FilePartition(2, partFiles.slice(4, 6)),
      FilePartition(3, Seq(partFiles(6)))
    )
    val oldset = new NVDataset(null, null, null, Some(oldPartitions))
    val newset = oldset.repartition(3)
    assertResult(3)(newset.partitions.length)
    for (i <- 0 until 3) {
      assertResult(i)(newset.partitions(i).index)
    }
    assert(newset.partitions(0).files.contains(partFiles(6)))
    assert(newset.partitions(1).files.contains(partFiles(5)))
    assert(newset.partitions(2).files.contains(partFiles(2)))
    assert(newset.partitions(2).files.contains(partFiles(1)))
    assert(newset.partitions(1).files.contains(partFiles(4)))
    assert(newset.partitions(2).files.contains(partFiles(3)))
    assert(newset.partitions(2).files.contains(partFiles(0)))
  }

  test("repartition more than number of files") {
    val oldPartitions = Seq(
      FilePartition(0, Seq(PartitionedFile(null, "/a/b/c", 0, 123))),
      FilePartition(1, Seq(PartitionedFile(null, "/a/b/d", 0, 456)))
    )
    val nvds = new NVDataset(null, null, null, Some(oldPartitions))
    assertThrows[UnsupportedOperationException] { nvds.repartition(3) }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
