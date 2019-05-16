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
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class NVDatasetSuite extends FunSuite with BeforeAndAfterAll {
  private final lazy val TRAIN_DATA_PATH = getTestDataPath("/rank.train.csv")
  private val spark = SparkSession.builder.master("local").getOrCreate()

  override protected def afterAll(): Unit = spark.close()

  test("mapColumnarSingleBatchPerPartition") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new NVDataReader(spark)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(TRAIN_DATA_PATH)
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
      reader.option("something", "something").csv(TRAIN_DATA_PATH)
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

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
