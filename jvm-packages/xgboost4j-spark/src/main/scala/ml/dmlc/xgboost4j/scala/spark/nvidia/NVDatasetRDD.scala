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

import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import org.apache.parquet.io.ParquetDecodingException
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile, SchemaColumnConvertNotSupportedException}
import org.apache.spark.sql.execution.QueryExecutionException

private[spark] class NVDatasetRDD(
  @transient private val sparkSession: SparkSession,
  readFunction: PartitionedFile => Iterator[NVColumnBatch] with AutoCloseable,
  @transient val filePartitions: Seq[FilePartition])
  extends RDD[NVColumnBatch](sparkSession.sparkContext, Nil) {

  override def compute(split: Partition, context: TaskContext): Iterator[NVColumnBatch] = {
    val iterator = new Iterator[NVColumnBatch] with AutoCloseable {
      private[this] val files = split.asInstanceOf[FilePartition].files.toIterator
      private[this] var currentFile: PartitionedFile = null
      private[this] var currentIterator: Iterator[NVColumnBatch] with AutoCloseable = null

      override def hasNext: Boolean = {
        (currentIterator != null && currentIterator.hasNext) || nextIterator()
      }

      override def next(): NVColumnBatch = currentIterator.next()

      private def readCurrentFile(): Iterator[NVColumnBatch] with AutoCloseable = {
        try {
          readFunction(currentFile)
        } catch {
          case e: FileNotFoundException =>
            throw new FileNotFoundException(
              e.getMessage + "\n" +
                "It is possible the underlying files have been updated. " +
                "You can explicitly invalidate the cache in Spark by " +
                "running 'REFRESH TABLE tableName' command in SQL or " +
                "by recreating the Dataset/DataFrame involved.")
        }
      }

      /** Advances to the next file. Returns true if a new non-empty iterator is available. */
      private def nextIterator(): Boolean = {
        if (files.hasNext) {
          currentFile = files.next()
          logInfo(s"Reading File $currentFile")
          currentIterator = readCurrentFile()
          try {
            hasNext
          } catch {
            case e: SchemaColumnConvertNotSupportedException =>
              val message = "Parquet column cannot be converted in " +
                s"file ${currentFile.filePath}. Column: ${e.getColumn}, " +
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
        } else {
          currentFile = null
          false
        }
      }

      override def close(): Unit = {
        if (currentIterator != null) {
          currentIterator.close();
        }
      }
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
