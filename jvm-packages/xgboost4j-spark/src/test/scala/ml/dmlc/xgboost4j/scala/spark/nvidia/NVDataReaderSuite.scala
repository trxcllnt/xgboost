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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.HadoopFsRelation
import org.apache.spark.sql.types.{BooleanType, DoubleType, IntegerType, StructType}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class NVDataReaderSuite extends FunSuite with BeforeAndAfterAll {
  private lazy val RANK_TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val RANK_TRAIN_PARQUET_PATH = getTestDataPath("/rank.train.parquet")
  private val spark = SparkSession.builder.master("local").getOrCreate()

  override protected def afterAll(): Unit = spark.close()

  test("csv parsing with DDL schema") {
    val reader = new NVDataReaderForTest(spark)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(RANK_TRAIN_CSV_PATH)
    assert(dataset != null)
    rankTrainCsvAssertions(reader)
  }

  test("csv parsing with StructType schema") {
    val reader = new NVDataReaderForTest(spark)
    val csvSchema = new StructType()
      .add("a", BooleanType)
      .add("b", DoubleType)
      .add("c", DoubleType)
      .add("d", DoubleType)
      .add("e", IntegerType)
    val dataset = reader.schema(csvSchema).csv(RANK_TRAIN_CSV_PATH)
    assert(dataset != null)
    rankTrainCsvAssertions(reader)
  }

  test("csv parsing with format/load") {
    val reader = new NVDataReaderForTest(spark)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.format("CSV").schema(csvSchema).load(RANK_TRAIN_CSV_PATH)
    assert(dataset != null)
    rankTrainCsvAssertions(reader)
  }

  test("parquet parsing with DDL schema") {
    val reader = new NVDataReaderForTest(spark)
    val specSchema = "a BOOLEAN, c DOUBLE, e INT"
    val dataset = reader.schema(specSchema).parquet(RANK_TRAIN_PARQUET_PATH)
    assert(dataset != null)
    assertResult("parquet") { reader.savedSourceType }
    assertResult(1) { reader.savedRelation.inputFiles.length }
    val schema = reader.savedRelation.schema
    assertResult(3) { schema.length }
    assertResult(BooleanType) { schema.fields(0).dataType }
    assertResult("a") { schema.fields(0).name }
    assertResult(DoubleType) { schema.fields(1).dataType }
    assertResult("c") { schema.fields(1).name }
    assertResult(IntegerType) { schema.fields(2).dataType }
    assertResult("e") { schema.fields(2).name }
  }

  test("parquet parsing with StructType schema") {
    val reader = new NVDataReaderForTest(spark)
    val specSchema = new StructType()
      .add("a", BooleanType)
      .add("e", IntegerType)
    val dataset = reader.schema(specSchema).parquet(RANK_TRAIN_PARQUET_PATH)
    assert(dataset != null)
    assertResult("parquet") { reader.savedSourceType }
    assertResult(1) { reader.savedRelation.inputFiles.length }
    val schema = reader.savedRelation.schema
    assertResult(2) { schema.length }
    assertResult(BooleanType) { schema.fields(0).dataType }
    assertResult("a") { schema.fields(0).name }
    assertResult(IntegerType) { schema.fields(1).dataType }
    assertResult("e") { schema.fields(1).name }
  }

  test("parquet parsing with format/load") {
    val reader = new NVDataReaderForTest(spark)
    val dataset = reader.format("parquet").load(RANK_TRAIN_PARQUET_PATH)
    assert(dataset != null)
    assertResult("parquet") { reader.savedSourceType }
    assertRankTrainLoad(reader)
  }

  test("invalid format type specified") {
    val reader = new NVDataReader(spark)
    assertThrows[UnsupportedOperationException] {
      reader.format("badformat").load(RANK_TRAIN_CSV_PATH)
    }
  }

  private def rankTrainCsvAssertions(reader: NVDataReaderForTest): Unit = {
    assertResult("csv") { reader.savedSourceType }
    assertRankTrainLoad(reader)
  }

  private def assertRankTrainLoad(reader: NVDataReaderForTest): Unit = {
    val schema = reader.savedRelation.schema
    assertResult(1) { reader.savedRelation.inputFiles.length }
    assertResult(5) { schema.length }
    assertResult(BooleanType) { schema.fields(0).dataType }
    assertResult("a") { schema.fields(0).name }
    assertResult(DoubleType) { schema.fields(1).dataType }
    assertResult("b") { schema.fields(1).name }
    assertResult(DoubleType) { schema.fields(2).dataType }
    assertResult("c") { schema.fields(2).name }
    assertResult(DoubleType) { schema.fields(3).dataType }
    assertResult("d") { schema.fields(3).name }
    assertResult(IntegerType) { schema.fields(4).dataType }
    assertResult("e") { schema.fields(4).name }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}

class NVDataReaderForTest(sparkSession: SparkSession) extends NVDataReader(sparkSession) {
  var savedRelation: HadoopFsRelation = _
  var savedSourceType: String = _
  var savedSourceOptions: Map[String, String] = _

  override protected def createDataset(relation: HadoopFsRelation,
    sourceType: String, sourceOptions: Map[String, String]): NVDataset = {
    savedRelation = relation
    savedSourceType = sourceType
    savedSourceOptions = sourceOptions
    super.createDataset(relation, sourceType, sourceOptions)
  }
}
