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

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{DataSource, HadoopFsRelation}

import scala.collection.JavaConverters._

class NVDataReader(sparkSession: SparkSession) {

  /**
    * Specifies the input data source format.
    */
  def format(source: String): NVDataReader = {
    this.source = source
    this
  }

  /**
    * Specifies the input schema to use when parsing data.
    */
  def schema(schema: StructType): NVDataReader = {
    specifiedSchema = Option(schema)
    this
  }

  /**
    * Specifies the schema by using the input DDL-formatted string.
    *
    * {{{
    *   spark.read.schema("a INT, b STRING, c DOUBLE").csv("test.csv")
    * }}}
    */
  def schema(schemaString: String): NVDataReader = {
    this.specifiedSchema = Option(StructType.fromDDL(schemaString))
    this
  }

  /**
    * Adds an input option for the underlying data source.
    */
  def option(key: String, value: String): NVDataReader = {
    extraOptions += (key -> value)
    this
  }

  /**
    * Adds an input option for the underlying data source.
    */
  def option(key: String, value: Boolean): NVDataReader = option(key, value.toString)

  /**
    * Adds an input option for the underlying data source.
    */
  def option(key: String, value: Long): NVDataReader = option(key, value.toString)

  /**
    * Adds an input option for the underlying data source.
    */
  def option(key: String, value: Double): NVDataReader = option(key, value.toString)

  /**
    * (Scala-specific) Adds input options for the underlying data source.
    */
  def options(options: scala.collection.Map[String, String]): NVDataReader = {
    extraOptions ++= options
    this
  }

  /**
    * Adds input options for the underlying data source.
    */
  def options(options: java.util.Map[String, String]): NVDataReader = {
    this.options(options.asScala)
    this
  }

  /**
    * Loads input in as an `NVDataset`, for data sources that don't require a path (e.g. external
    * key-value stores).
    */
  def load(): NVDataset = {
    load(Seq.empty: _*) // force invocation of `load(...varargs...)`
  }

  /**
    * Loads input in as an `NVDataset`, for data sources that require a path (e.g. data backed by
    * a local or distributed file system).
    */
  def load(path: String): NVDataset = {
    // force invocation of `load(...varargs...)`
    option("path", path).load(Seq.empty: _*)
  }

  /**
    * Loads input in as an `NVDataset`, for data sources that support multiple paths.
    * Only works if the source is a HadoopFsRelationProvider.
    */
  @scala.annotation.varargs
  def load(paths: String*): NVDataset = {
    val sourceType = sourceTypeMap.getOrElse(source.toLowerCase(),
      throw new UnsupportedOperationException("Unsupported input format: " + source))
    val optionsMap = extraOptions.toMap
    val fsRelation = DataSource.apply(sparkSession,
      paths = paths,
      userSpecifiedSchema = specifiedSchema,
      className = source,
      options = optionsMap).resolveRelation()
    createDataset(fsRelation.asInstanceOf[HadoopFsRelation], sourceType, optionsMap)
  }

  protected def createDataset(relation: HadoopFsRelation, sourceType: String,
      sourceOptions: Map[String, String]): NVDataset = {
    new NVDataset(relation, sourceType, sourceOptions)
  }

  /**
    * Loads a CSV file and returns the result as an `NVDataset`. See the documentation on the
    * other overloaded `csv()` method for more details.
    */
  def csv(path: String): NVDataset = {
    // This method ensures that calls that explicit need single argument works, see SPARK-16009
    csv(Seq(path): _*)
  }

  /**
    * Loads CSV files and returns the result as an `NVDataset`.
    * You can set the following CSV-specific options to deal with CSV files:
    * <ul>
    * <li>`sep` (default `,`): sets a single character as a separator for each
    * field and value.</li>
    * <li>`quote` (default `"`): sets a single character used for escaping quoted values where
    * the separator can be part of the value.</li>
    * <li>`comment` (default empty string): sets a single character used for skipping lines
    * beginning with this character. By default, it is disabled.</li>
    * <li>`header` (default `false`): uses the first line as names of columns.</li>
    * <li>`nullValue` (default empty string): sets the string representation of a null value.</li>
    * </ul>
    */
  @scala.annotation.varargs
  def csv(paths: String*): NVDataset = format("csv").load(paths : _*)

  /**
    * Loads a Parquet file, returning the result as an `NVDataset`. See the documentation
    * on the other overloaded `parquet()` method for more details.
    */
  def parquet(path: String): NVDataset = {
    // This method ensures that calls that explicit need single argument works, see SPARK-16009
    parquet(Seq(path): _*)
  }

  /**
    * Loads a Parquet file, returning the result as an `NVDataset`.
    *
    * You can set the following Parquet-specific option(s) for reading Parquet files:
    * <ul>
    * <li>`mergeSchema` (default is the value specified in `spark.sql.parquet.mergeSchema`): sets
    * whether we should merge schemas collected from all Parquet part-files. This will override
    * `spark.sql.parquet.mergeSchema`.</li>
    * </ul>
    */
  @scala.annotation.varargs
  def parquet(paths: String*): NVDataset = {
    format("parquet").load(paths: _*)
  }

  private var source: String = sparkSession.sessionState.conf.defaultDataSourceName

  private var specifiedSchema: Option[StructType] = None

  private val extraOptions = new scala.collection.mutable.HashMap[String, String]

  private val sourceTypeMap: Map[String, String] = Map(
    "csv" -> "csv",
    "com.databricks.spark.csv" -> "csv",
    "parquet" -> "parquet",
    "org.apache.spark.sql.parquet" -> "parquet",
    "org.apache.spark.sql.parquet.DefaultSource" -> "parquet",
    "org.apache.spark.sql.execution.datasources.parquet" -> "parquet",
    "org.apache.spark.sql.execution.datasources.parquet.DefaultSource" -> "parquet"
  )
}

object NVDataReader {
  def apply(spark: SparkSession): NVDataReader = new NVDataReader(spark)
}
