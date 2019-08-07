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

import java.io.FileNotFoundException
import java.util.NoSuchElementException

import ai.rapids.cudf.Table
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.io.ParquetDecodingException
import org.apache.spark.{Partition, SerializableWritable, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile, SchemaColumnConvertNotSupportedException}
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.types.StructType

import scala.collection.mutable.ArrayBuffer

private[spark] class GpuDatasetRDD(
    @transient private val sparkSession: SparkSession,
    broadcastedConf: Broadcast[SerializableWritable[Configuration]],
    readFunction: (Configuration, PartitionedFile) => Option[Table with AutoCloseable],
    @transient val filePartitions: Seq[FilePartition],
    schema: StructType)
  extends RDD[GpuColumnBatch](sparkSession.sparkContext, Nil) {

  private val logger = LogFactory.getLog(classOf[GpuDatasetRDD])

  // The resulting iterator will only return a single GpuColumnBatch.
  override def compute(split: Partition, context: TaskContext): Iterator[GpuColumnBatch] = {
    val conf: Configuration = broadcastedConf.value.value
    val iterator = new Iterator[GpuColumnBatch] with AutoCloseable {
      private[this] var batchedTable = GpuDatasetRDD.buildBatch(conf, readFunction,
          split.asInstanceOf[FilePartition].files)
      private[this] var resultBatch: Option[GpuColumnBatch] =
          batchedTable.map({ new GpuColumnBatch(_, schema) })

      override def hasNext: Boolean = resultBatch.isDefined

      override def next(): GpuColumnBatch = {
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
      try {
        file.locations.filter(_ != "localhost").foreach { host =>
          hostToNumBytes(host) = hostToNumBytes.getOrElse(host, 0L) + file.length
        }
      } catch {
        case e: NoSuchMethodError =>
          // This would fail because some Spark implementations have overloaded FilePartition
          // We handle this in GpuDataset but we never set the locations in those cases
          // so do nothing here
          logger.debug("Not a Spark FilePartition, skipping getting locations")
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

private[spark] object GpuDatasetRDD {

  private def buildBatch(conf: Configuration,
                         readFunc: (Configuration, PartitionedFile) => Option[Table],
                         partfiles: Seq[PartitionedFile]): Option[Table] = {
    var result: Option[Table] = None
    if (partfiles.isEmpty) {
      return result
    }

    val tables = ArrayBuffer[Table]()
    try {
      for ((partfile, i) <- partfiles.zipWithIndex) {
        readPartFile(conf, readFunc, partfile) match {
          case Some(table) => {
            tables.append(table)
          }
          case None => {
            // do nothing
          }
        }
      }
      result = if (tables.length > 1) {
        Some(Table.concatenate(tables: _*))
      } else if (tables.length == 1){
        Some(tables.remove(0))
      } else {
        None
      }
    } finally {
      tables.foreach(_.close())
    }
    result
  }

  private def readPartFile(
      conf: Configuration,
      readFunc: (Configuration, PartitionedFile) => Option[Table],
      partfile: PartitionedFile): Option[Table] = {
    try {
      readFunc(conf, partfile)
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
