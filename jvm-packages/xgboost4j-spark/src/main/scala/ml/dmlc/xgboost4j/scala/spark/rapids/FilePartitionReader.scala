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

import java.io.{FileNotFoundException, IOException}

import ml.dmlc.xgboost4j.java.spark.rapids.PartitionReader
import org.apache.parquet.io.ParquetDecodingException
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.execution.datasources.SchemaColumnConvertNotSupportedException
import org.apache.spark.sql.internal.SQLConf

class FilePartitionReader[T](readers: Iterator[PartitionedFileReader[T]])
  extends PartitionReader[T] with Logging {
  private var currentReader: PartitionedFileReader[T] = null

  private val sqlConf = SQLConf.get

  private def ignoreMissingFiles = sqlConf.ignoreMissingFiles

  private def ignoreCorruptFiles = sqlConf.ignoreCorruptFiles

  override def next(): Boolean = {
    if (currentReader == null) {
      if (readers.hasNext) {
        try {
          currentReader = getNextReader()
        } catch {
          case e: FileNotFoundException if ignoreMissingFiles =>
            logWarning(s"Skipped missing file: $currentReader", e)
            currentReader = null
            return false
          // Throw FileNotFoundException even if `ignoreCorruptFiles` is true
          case e: FileNotFoundException if !ignoreMissingFiles =>
            throw new FileNotFoundException(
              e.getMessage + "\n" +
                "It is possible the underlying files have been updated. " +
                "You can explicitly invalidate the cache in Spark by " +
                "recreating the Dataset/DataFrame involved.")
          case e@(_: RuntimeException | _: IOException) if ignoreCorruptFiles =>
            logWarning(
              s"Skipped the rest of the content in the corrupted file: $currentReader", e)
            currentReader = null
            return false
        }
      } else {
        return false
      }
    }

    // In PartitionReader.next(), the current reader proceeds to next record.
    // It might throw RuntimeException/IOException and Spark should handle these exceptions.
    val hasNext = try {
      currentReader.next()
    } catch {
      case e: SchemaColumnConvertNotSupportedException =>
        val message = "Parquet column cannot be converted in " +
          s"file ${currentReader.file.filePath}. Column: ${e.getColumn}, " +
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
      case e@(_: RuntimeException | _: IOException) if ignoreCorruptFiles =>
        logWarning(
          s"Skipped the rest of the content in the corrupted file: $currentReader", e)
        false
    }
    if (hasNext) {
      true
    } else {
      close()
      currentReader = null
      next()
    }
  }

  override def get(): T = currentReader.get()

  override def close(): Unit = {
    if (currentReader != null) {
      currentReader.close()
    }
  }

  private def getNextReader(): PartitionedFileReader[T] = {
    val reader = readers.next()
    logInfo(s"Reading file $reader")
    val file = reader.file
    reader
  }
}
