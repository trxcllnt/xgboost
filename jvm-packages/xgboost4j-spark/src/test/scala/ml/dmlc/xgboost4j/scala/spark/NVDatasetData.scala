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

package ml.dmlc.xgboost4j.scala.spark

import java.io.File

import ai.rapids.cudf.{DType, Table}
import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import ml.dmlc.xgboost4j.scala.spark.nvidia.{NVDataReader, NVDataset}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{
  BooleanType, ByteType, DataType, DateType, DoubleType,
  FloatType, IntegerType, LongType, ShortType, StructField, StructType, TimestampType
}

object NVDatasetData {

  private val classifierTrainDataPath = "/iris.data.csv"
  private val classifierTestDataPath = "/iris.test.data.csv"

  private val classifierSchema = new StructType(Array(
    StructField("sepal length", FloatType),
    StructField("sepal width", FloatType),
    StructField("petal length", FloatType),
    StructField("petal width", FloatType),
    StructField("classIndex", FloatType)))

  private val regressionTrainDataPath = "/norank.train.csv"
  private val regressionTestDataPath = "/norank.train.csv"
  private val regressionSchema = new StructType(Array(
    StructField("b", DoubleType),
    StructField("c", DoubleType),
    StructField("d", DoubleType),
    StructField("e", DoubleType)))

  private def toDType(dataType: DataType): DType = {
    dataType match {
      case BooleanType | ByteType => DType.INT8
      case ShortType => DType.INT16
      case IntegerType => DType.INT32
      case LongType => DType.INT64
      case FloatType => DType.FLOAT32
      case DoubleType => DType.FLOAT64
      case DateType => DType.DATE32
      case TimestampType => DType.TIMESTAMP
      case unknownType => throw new UnsupportedOperationException(
        s"Unsupported Spark SQL type $unknownType")
    }
  }

  def getClassifierTrainNVDataset(spark: SparkSession): NVDataset = {
    val reader = new NVDataReader(spark)
    reader.schema(classifierSchema).csv(getTestDataPath(classifierTrainDataPath))
  }

  def getClassifierTestNVDataset(spark: SparkSession): (NVDataset, Long) = {
    val reader = new NVDataReader(spark)
    val test = reader.schema(classifierSchema).csv(getTestDataPath(classifierTestDataPath))

    val counts = test.mapColumnarSingleBatchPerPartition(nvColumnBatch => {
      Iterator.single(nvColumnBatch.getNumRows)
    }).collect()

    (test, counts(0))
  }

  def getRegressionTrainNVDataset(spark: SparkSession): NVDataset = {
    val reader = new NVDataReader(spark)
    reader.schema(regressionSchema).csv(getTestDataPath(regressionTrainDataPath))
  }

  def getRegressionTestNVDataset(spark: SparkSession): (NVDataset, Long) = {
    val reader = new NVDataReader(spark)
    val test = reader.schema(regressionSchema).csv(getTestDataPath(regressionTestDataPath))

    val counts = test.mapColumnarSingleBatchPerPartition(nvColumnBatch => {
      Iterator.single(nvColumnBatch.getNumRows)
    }).collect()

    (test, counts(0))
  }

  private def getColumns(file: String, schema: StructType, label: String):
  (Table, Array[Long], Array[Long]) = {
    val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
    schema.foreach(f => csvSchemaBuilder.column(toDType(f.dataType), f.name))
    val table = Table.readCSV(csvSchemaBuilder.build(), new File(getTestDataPath(file)))
    val nvColumnBatch = new NVColumnBatch(table, schema)

    val featuresColNames = schema.fieldNames.filter(_ != label)
    val featuresHandle = featuresColNames.map(colName => {
      try {
        nvColumnBatch.getColumn(schema.fieldIndex(colName))
      } catch {
        case e: Exception => -1L
      }
    }).filter(_ > 0)

    val labelsHandle = Array(label).map(colName => {
      try {
        nvColumnBatch.getColumn(schema.fieldIndex(colName))
      } catch {
        case e: Exception => -1L
      }
    }).filter(_ > 0)

    (table, featuresHandle, labelsHandle)
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

  def classifierCleanUp(): Unit = {
    if (classifierTrainTable != null) {
      classifierTrainTable.close()
      classifierTrainTable = null
    }

    if (classifierTestTable != null) {
      classifierTestTable.close()
      classifierTestTable = null
    }
  }

  def regressionCleanUp(): Unit = {
    if (regressionTestTable != null) {
      regressionTestTable.close()
      regressionTestTable = null
    }

    if (regressionTrainTable != null) {
      regressionTrainTable.close()
      regressionTrainTable = null
    }
  }

  var classifierTestTable: Table = null
  var classifierTrainTable: Table = null

  lazy val classifierTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTestDataPath,
      classifierSchema, "classIndex")
    classifierTestTable = table;
    (features, labels)
  }
  lazy val classifierTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTrainDataPath,
      classifierSchema, "classIndex")
    classifierTrainTable = table
    (features, labels)
  }
  lazy val classifierFeatureCols: Seq[String] =
    classifierSchema.fieldNames.filter(_ != "classIndex")


  var regressionTestTable: Table = null
  var regressionTrainTable: Table = null

  lazy val regressionTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTestDataPath,
      regressionSchema, "e")
    regressionTestTable = table
    (features, labels)
  }
  lazy val regressionTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTrainDataPath,
      regressionSchema, "e")
    regressionTrainTable = table
    (features, labels)
  }

  lazy val regressionFeatureCols: Seq[String] =
    regressionSchema.fieldNames.filter(_ != "e")
}

