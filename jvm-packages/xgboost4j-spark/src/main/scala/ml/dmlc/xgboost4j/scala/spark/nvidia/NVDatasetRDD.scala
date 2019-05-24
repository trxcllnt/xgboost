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

import java.io.FileNotFoundException
import java.util.NoSuchElementException

import ai.rapids.cudf.Table
import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import org.apache.parquet.io.ParquetDecodingException
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile, SchemaColumnConvertNotSupportedException}
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.types.StructType

private[spark] class NVDatasetRDD(
    @transient private val sparkSession: SparkSession,
    readFunction: PartitionedFile => Table with AutoCloseable,
    @transient val filePartitions: Seq[FilePartition],
    schema: StructType)
  extends RDD[NVColumnBatch](sparkSession.sparkContext, Nil) {

  // The resulting iterator will only return a single NVColumnBatch.
  override def compute(split: Partition, context: TaskContext): Iterator[NVColumnBatch] = {
    val iterator = new Iterator[NVColumnBatch] with AutoCloseable {
      private[this] var batchedTable = NVDatasetRDD.buildBatch(readFunction,
          split.asInstanceOf[FilePartition].files)
      private[this] var resultBatch: Option[NVColumnBatch] =
          batchedTable.map({ new NVColumnBatch(_, schema) })

      override def hasNext: Boolean = resultBatch.isDefined

      override def next(): NVColumnBatch = {
        val batch = resultBatch.getOrElse(throw new NoSuchElementException)
        resultBatch = None
        batch
      }

      override def close(): Unit = batchedTable.foreach({ _.close() })

    }

    // Register an on-task-completion callback to close the input stream.
    context.addTaskCompletionListener(_ => iterator.close())
    iterator
  }

  override protected def getPartitions: Array[Partition] = filePartitions.toArray

  override protected def getPreferredLocations(split: Partition): Seq[String] = {
    val filePart = split.asInstanceOf[FilePartition]
    val hostToNumBytes = scala.collection.mutable.HashMap.empty[String, Long]
    filePart.files.foreach { file =>
      file.locations.filter(_ != "localhost").foreach { host =>
        hostToNumBytes(host) = hostToNumBytes.getOrElse(host, 0L) + file.length
      }
    }

    // Takes the first 3 hosts with the most data to be retrieved
    hostToNumBytes.toSeq.sortBy {
      case (host, numBytes) => numBytes
    }.reverse.take(3).map {
      case (host, numBytes) => host
    }
  }
}

private[spark] object NVDatasetRDD {

  private def buildBatch(readFunc: PartitionedFile => Table,
      partfiles: Seq[PartitionedFile]): Option[Table] = {
    var result: Option[Table] = None
    if (partfiles.isEmpty) {
      return result
    }

    val tables = new Array[Table](partfiles.length)
    val haveMultipleTables = tables.length > 1
    try {
      for ((partfile, i) <- partfiles.zipWithIndex) {
        tables(i) = readPartFile(readFunc, partfile)
      }
      result = Some(if (haveMultipleTables) Table.concatenate(tables: _*) else tables(0))
    } finally {
      if (haveMultipleTables) {
        for (table <- tables if table != null) {
          table.close()
        }
      }
    }

    result
  }

  private def readPartFile(readFunc: PartitionedFile => Table, partfile: PartitionedFile): Table = {
    try {
      readFunc(partfile)
    } catch {
      case e: FileNotFoundException =>
        throw new FileNotFoundException(
          e.getMessage + "\n" +
            "It is possible the underlying files have been updated. " +
            "You can explicitly invalidate the cache in Spark by " +
            "running 'REFRESH TABLE tableName' command in SQL or " +
            "by recreating the Dataset/DataFrame involved.")
      case e: SchemaColumnConvertNotSupportedException =>
        val message = "Parquet column cannot be converted in " +
            s"file ${partfile.filePath}. Column: ${e.getColumn}, " +
            s"Expected: ${e.getLogicalType}, Found: ${e.getPhysicalType}"
        throw new QueryExecutionException(message, e)
      case e: ParquetDecodingException =>
        if (e.getMessage.contains("Can not read value at")) {
          val message = "Encounter error while reading parquet files. " +
              "One possible cause: Parquet column cannot be converted in the " +
              "corresponding files. Details: "
          throw new QueryExecutionException(message, e)
        }
        throw e
    }
  }
}
