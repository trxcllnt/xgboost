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
import org.apache.spark.sql.{Row, SparkSession}
import org.scalactic.TolerantNumerics
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class NVDatasetSuite extends FunSuite with BeforeAndAfterAll {
  private lazy val TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val TRAIN_PARQUET_PATH = getTestDataPath("/rank.train.parquet")
  private val spark = SparkSession.builder.master("local").getOrCreate()

  override protected def afterAll(): Unit = spark.close()

  test("mapColumnarSingleBatchPerPartition") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(spark)
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
    val reader = new NVDataReader(spark)
    assertThrows[UnsupportedOperationException] {
      reader.option("something", "something").csv(TRAIN_CSV_PATH)
        .mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
          Iterator.single(b.getNumColumns))
    }
  }

  test("CSV parsing with header") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(spark)
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
    val reader = new NVDataReader(spark)
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
    val reader = new NVDataReader(spark)
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

  test("Parquet parsing") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(spark)
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
    val reader = new NVDataReader(spark)
    val specSchema = "a BOOLEAN, c DOUBLE, e INT"
    val dataset = reader.schema(specSchema).parquet(TRAIN_PARQUET_PATH)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single((b.getNumColumns, b.getNumRows)))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(3) { counts(0)._1 }
    assertResult(149) { counts(0)._2 }
  }

  test("zipPartitionsAsRows") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(spark)
    val dataPath = getTestDataPath("/rank.train.csv")
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(dataPath)
    val rddOther = spark.sparkContext.parallelize(1 to 1000, 1)
    val zipFunc = (rowIter: Iterator[Row], numIter: Iterator[Int]) => new Iterator[(Int, Row)] {
      override def hasNext: Boolean = rowIter.hasNext

      override def next(): (Int, Row) = (numIter.next, rowIter.next)
    }

    val rddZipped = dataset.zipPartitionsAsRows(rddOther, true)(zipFunc)
    val data = rddZipped.collect

    assertResult(149) { data.length }
    assertResult(1) { data(0)._1 }
    val firstRow = data(0)._2
    assertResult(5) { firstRow.size }
    assertResult(false) { firstRow.getBoolean(0) }
    implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.000000001)
    assert(firstRow.getDouble(1) === 985.574005058)
    assert(firstRow.getDouble(2) === 320.223538037)
    assert(firstRow.getDouble(3) === 0.621236086198)
    assertResult(1) { firstRow.get(4) }

    assertResult(149) { data(148)._1 }
    val lastRow = data(148)._2
    assertResult(5) { lastRow.size }
    assertResult(false) { firstRow.getBoolean(0) }
    assert(lastRow.getDouble(1) === 9.90744703306)
    assert(lastRow.getDouble(2) === 50.810461183)
    assert(lastRow.getDouble(3) === 3.00863325197)
    assertResult(20) { lastRow.get(4) }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
