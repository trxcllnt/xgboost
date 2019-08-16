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

import java.util.Locale
import java.io.OutputStream
import java.net.URI
import java.nio.charset.StandardCharsets
import java.nio.file.FileAlreadyExistsException
import java.util.Collections

import ai.rapids.cudf.{CSVOptions, ColumnVector, DType, HostMemoryBuffer, ORCOptions, ParquetOptions, Table}
import ml.dmlc.xgboost4j.java.XGBoostSparkJNI
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import org.apache.commons.logging.LogFactory
import org.apache.commons.io.output.{CountingOutputStream, NullOutputStream}
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{BlockLocation, FSDataInputStream, FSDataOutputStream, FileStatus, LocatedFileStatus, Path}
import org.apache.parquet.bytes.BytesUtils
import org.apache.parquet.format.converter.ParquetMetadataConverter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.{BlockMetaData, ColumnChunkMetaData, ColumnPath, FileMetaData, ParquetMetadata}
import org.apache.parquet.schema.MessageType
import org.apache.spark.{SerializableWritable, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.execution.datasources.{FilePartition, HadoopFileLinesReader, HadoopFsRelation, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.Platform

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.Random


/*
 * Note on compatibility with different versions of Spark:
 *
 * Some versions of Spark have slightly different internal classes. The list of
 * known ones are below. If any of these are used make sure we handle appropriately.
 *
 * PartitionDirectory:
 *   Apache Spark:
 *     case class PartitionDirectory(values: InternalRow, files: Seq[FileStatus])
 *   We have seen it overloaded as:
 *     case class PartitionDirectory(values: InternalRow, files: Seq[OtherFileStatus])
 *       Where OtherFileStatus is not a Hadoop FileStatus but has the same methods for
 *       getLen and getPath
 * FilePartition:
 *   Apache Spark:
 *     case class FilePartition(index: Int, files: Array[PartitionedFile])
 *   We have see this overloaded as:
 *     case class FilePartition(index: Int, files: Seq[PartitionedFile],
 *       preferredHosts: Seq[String])
 * PartitionedFile:
 *   Apache Spark:
 *     case class PartitionedFile(partitionValues: InternalRow,
 *       filePath: String, start: Long, length: Long, locations: Array[String])
 *   We have seen it overloaded as:
 *     case class PartitionedFile(partitionValues: InternalRow,
 *       filePath: String, start: Long, length: Long, locations: Seq[String])
 */

class GpuDataset(fsRelation: HadoopFsRelation,
    sourceType: String,
    sourceOptions: Map[String, String],
    castAllToFloats: Boolean,
    specifiedPartitions: Option[Seq[FilePartition]] = None) {

  private val logger = LogFactory.getLog(classOf[GpuDataset])


  /** Returns the schema of the data. */
  def schema: StructType = {
    if (castAllToFloats) {
      GpuDataset.numericAsFloats(fsRelation.schema)
    } else {
      fsRelation.schema
    }
  }

  /**
    *  Return an [[RDD]] of column batches.
    *  NOTE: GpuColumnBatch is NOT serializable, so one has to be very careful in
    *        the types of operations performed on this RDD!
    */
  private[xgboost4j] def buildRDD: RDD[GpuColumnBatch] = {
    val partitionReader = GpuDataset.getPartFileReader(fsRelation.sparkSession, sourceType,
      fsRelation.schema, sourceOptions, castAllToFloats)
    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val serializableConf = new SerializableWritable[Configuration](hadoopConf)
    val broadcastedConf = sparkSession.sparkContext.broadcast(serializableConf)
    new GpuDatasetRDD(fsRelation.sparkSession, broadcastedConf, partitionReader,
      partitions, schema)
  }

  /**
    * Return an [[RDD]] by applying a function expecting RAPIDS cuDF column pointers
    * to each partition of this data.
    */
  private[xgboost4j] def mapColumnarSingleBatchPerPartition[U: ClassTag](
      func: GpuColumnBatch => Iterator[U]): RDD[U] = {
    buildRDD.mapPartitions(GpuDataset.getMapper(func))
  }

  /** Return a new GpuDataset that has exactly numPartitions partitions. */
  def repartition(numPartitions: Int): GpuDataset = {
    if (numPartitions == partitions.length) {
      return new GpuDataset(fsRelation, sourceType, sourceOptions, castAllToFloats,
        Some(partitions))
    }

    // build a list of all input files sorted from largest to smallest
    var files = partitions.flatMap(_.files)
    var permitNumPartitions = numPartitions
    files = if (files.length < numPartitions) {
      val totalSize = files.map(_.length).sum
      var newPartitionSize = totalSize / numPartitions
      if (newPartitionSize < 1) {
        // to handle corner case
        newPartitionSize = 1
        permitNumPartitions = totalSize.toInt
      }
      val splitPartFiles = ArrayBuffer[PartitionedFile]()
      for (file <- files) {
        // make sure each partfile is smaller than newPartitionSize
        if (file.length > newPartitionSize) {
          splitPartFiles ++= splitFileWhenRepartition(file, newPartitionSize)
        } else {
          splitPartFiles += file
        }
      }
      splitPartFiles
    } else {
      files
    }.sortBy(_.length)(Ordering[Long].reverse)


    // Seed the partition buckets with one of the largest files then iterate
    // through the rest of the files, adding each to the smallest bucket
    val buckets = files.take(permitNumPartitions).map(new GpuDataset.PartBucket(_))
    def bucketOrder(b: GpuDataset.PartBucket) = -b.getSize
    val queue = mutable.PriorityQueue(buckets: _*)(Ordering.by(bucketOrder))
    for (file <- files.drop(permitNumPartitions)) {
      val bucket = queue.dequeue()
      bucket.addFile(file)
      queue.enqueue(bucket)
    }

    val newPartitions = buckets.zipWithIndex.map{case (b, i) => b.toFilePartition(i)}
    new GpuDataset(fsRelation, sourceType, sourceOptions, castAllToFloats, Some(newPartitions))
  }

  /**
    * Find the number of classes for the specified label column. The largest value within
    * the column is assumed to be one less than the number of classes.
    */
  def findNumClasses(labelCol: String): Int = {
    val fieldIndex = schema.fieldIndex(labelCol)
    val rdd = mapColumnarSingleBatchPerPartition(GpuDataset.maxDoubleMapper(fieldIndex))
    val maxVal = rdd.reduce(Math.max)
    val numClasses = maxVal + 1
    require(numClasses.isValidInt, s"Found max label value =" +
        s" $maxVal but requires integers in range [0, ... ${Int.MaxValue})")
    numClasses.toInt
  }

  /** The SparkSession that created this GpuDataset. */
  def sparkSession: SparkSession = fsRelation.sparkSession

  private def getTotalBytes(
      partitions: Seq[PartitionDirectory],
      openCostInBytes: Long): Long = {
    partitions.flatMap { partitionDir =>
      getFileLengthsFromPartitionDir(partitionDir, openCostInBytes)
    }.sum
  }

  private def getFileLengthsFromPartitionDir(
      partitionDir: PartitionDirectory,
      openCostInBytes: Long): Seq[Long] = {
    if (partitionDir.files.nonEmpty) {
      // Should be a Hadoop FileStatus, but we have seen people overload this
      // Luckily the overloaded version has the same getLen method. Here we
      // only do the reflection when we really need to.
      if (partitionDir.files(0).isInstanceOf[FileStatus]) {
        partitionDir.files.map(_.getLen + openCostInBytes)
      } else {
        logger.debug(s"Not a FileStatus, class is: ${partitionDir.files(0).getClass}")
        partitionDir.files.asInstanceOf[Seq[Any]].map { file =>
          getLenReflection(file) + openCostInBytes
        }.asInstanceOf[Seq[Long]]
      }
    } else {
      Seq.empty[Long]
    }
  }

  private def getLenReflection(file: Any): Long = {
    val len = try {
      val getLen = file.getClass.getMethod("getLen")
      getLen.invoke(file).asInstanceOf[Long]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported File Status type ${file.getClass}, failed calling getLen"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    len.asInstanceOf[Long]
  }

  private def getPathReflection(file: Any): Path = {
    val fpath = try {
      val getPath = file.getClass.getMethod("getPath")
      getPath.invoke(file).asInstanceOf[Path]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported File Status type ${file.getClass}, failed calling getPath"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    fpath.asInstanceOf[Path]
  }


  private def fileTypeSupportSplit(fileType: String): Boolean = {
    fileType match {
      case "csv" => true
      case "parquet" => true
      case _ => false
    }
  }

  private def getSplits(
      partitions: Seq[PartitionDirectory],
      maxSplitBytes: Long): Seq[PartitionedFile] = {
    partitions.flatMap { partitionDir =>
      if (partitionDir.files.nonEmpty) {
        // Should be a Hadoop FileStatus, but we have seen people overload this
        // Luckily the overloaded version has the same methods. Here we
        // only do the reflection when we really need to.
        if (partitionDir.files(0).isInstanceOf[FileStatus]) {
          partitionDir.files.filter(_.getLen > 0).flatMap { file =>
            // getPath() is very expensive so we only want to call it once in this block:
            val filePath = file.getPath
            val isSplitable = fsRelation.fileFormat.isSplitable(
              fsRelation.sparkSession, fsRelation.options, filePath) &&
              fileTypeSupportSplit(sourceType)
            splitFile(
              file = file,
              filePath = filePath,
              isSplitable = isSplitable,
              maxSplitBytes = maxSplitBytes,
              partitionValues = partitionDir.values
            )
          }
        } else {
          logger.debug(s"Not a FileStatus, class is: ${partitionDir.files(0).getClass}")
          val pfs = partitionDir.files.asInstanceOf[Seq[Any]].flatMap { file =>
            // skip the splitable stuff for now since not used anyway
            splitFileReflection(
              filePath = getPathReflection(file),
              fileLength = getLenReflection(file),
              partitionValues = partitionDir.values
            )
          }.asInstanceOf[Seq[PartitionedFile]]
          pfs.filter(_.length > 0)
        }
      } else {
        Seq.empty[PartitionedFile]
      }
    }.toArray.sortBy(_.length)(implicitly[Ordering[Long]].reverse)
  }

  private[rapids] lazy val partitions: Seq[FilePartition] = specifiedPartitions.getOrElse{
    val selectedPartitions = fsRelation.location.listFiles(Nil, Nil)
    val openCostInBytes = fsRelation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val totalBytes = getTotalBytes(selectedPartitions, openCostInBytes)
    val maxSplitBytes = computeMaxSplitBytes(fsRelation.sparkSession, totalBytes)
    logger.info(s"Planning scan with bin packing, max size: $maxSplitBytes bytes, " +
      s"open cost is considered as scanning $openCostInBytes bytes, " +
      s"total bytes is $totalBytes.")

    val splits = getSplits(selectedPartitions, maxSplitBytes)
    val partitions = getFilePartitions(fsRelation.sparkSession, splits, maxSplitBytes)
    partitions
  }

  private def getFilePartitions(
      sparkSession: SparkSession,
      partitionedFiles: Seq[PartitionedFile],
      maxSplitBytes: Long): Seq[FilePartition] = {
    val partitions = new ArrayBuffer[FilePartition]
    val currentFiles = new ArrayBuffer[PartitionedFile]
    var currentSize = 0L

    /** Close the current partition and move to the next. */
    def closePartition(): Unit = {
      if (currentFiles.nonEmpty) {
        // Copy to a new Array.
        val newPartition =
          GpuDataset.createFilePartition(partitions.size, currentFiles.toArray.toSeq)
        partitions += newPartition
      }
      currentFiles.clear()
      currentSize = 0
    }

    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    // Assign files to partitions using "Next Fit Decreasing"
    partitionedFiles.foreach { file =>
      if (currentSize + file.length > maxSplitBytes) {
        closePartition()
      }
      // Add the given file to the current partition.
      currentSize += file.length + openCostInBytes
      currentFiles += file
    }
    closePartition()
    partitions
  }

  private def computeMaxSplitBytes(sparkSession: SparkSession,
      totalBytes: Long): Long = {
    val defaultMaxSplitBytes = sparkSession.sessionState.conf.filesMaxPartitionBytes
    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    val defaultParallelism = sparkSession.sparkContext.defaultParallelism
    val bytesPerCore = totalBytes / defaultParallelism

    Math.min(defaultMaxSplitBytes, Math.max(openCostInBytes, bytesPerCore))
  }

  private def splitFileReflection(
      filePath: Path,
      fileLength: Long,
      partitionValues: InternalRow): Seq[PartitionedFile] = {
    // Apache Spark open source:
    // case class PartitionedFile(partitionValues: InternalRow,
    //   filePath: String, start: Long, length: Long, locations: Array[String])
    // We have seen it overloaded as:
    // case class PartitionedFile(partitionValues: InternalRow,
    //   filePath: String, start: Long, length: Long, locations: Seq[String])
    val partitionedFile = try {
      val partedFileApply = org.apache.spark.sql.execution.datasources.PartitionedFile.getClass
        .getMethod("apply", classOf[InternalRow], classOf[String], classOf[Long],
          classOf[Long], classOf[Seq[String]])

      // not getting the file locations for now
      partedFileApply.invoke(org.apache.spark.sql.execution.datasources.PartitionedFile,
        partitionValues, filePath.toUri.toString, new java.lang.Long(0),
          new java.lang.Long(fileLength), Seq.empty[String]).asInstanceOf[PartitionedFile]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported PartitionedFile type, failed to create"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    Seq(partitionedFile)
  }

  private def splitFile(
      file: FileStatus,
      filePath: Path,
      isSplitable: Boolean,
      maxSplitBytes: Long,
      partitionValues: InternalRow): Seq[PartitionedFile] = {

    if (isSplitable) {
      (0L until file.getLen by maxSplitBytes).map { offset =>
        val remaining = file.getLen - offset
        val size = if (remaining > maxSplitBytes) maxSplitBytes else remaining
        val hosts = getBlockHosts(getBlockLocations(file), offset, size)
        PartitionedFile(partitionValues, filePath.toUri.toString, offset, size, hosts)
      }
    } else {
      Seq(getPartitionedFile(file, filePath, partitionValues))
    }
  }

  private def splitFileWhenRepartition(partFile: PartitionedFile,
                                        splitBytes: Long
                                      ): Seq[PartitionedFile] = {
    val partFileEnd = partFile.start + partFile.length
    (partFile.start until partFileEnd by splitBytes).map { offset =>
      val size = Math.min(partFileEnd - offset, splitBytes)
      PartitionedFile(partFile.partitionValues, partFile.filePath, offset, size, partFile.locations)
    }
  }

  private def getPartitionedFile(
      file: FileStatus,
      filePath: Path,
      partitionValues: InternalRow): PartitionedFile = {
    val hosts = getBlockHosts(getBlockLocations(file), 0, file.getLen)
    PartitionedFile(partitionValues, filePath.toUri.toString, 0, file.getLen, hosts)
  }

  private def getBlockLocations(file: FileStatus): Array[BlockLocation] = file match {
    case f: LocatedFileStatus => f.getBlockLocations
    case _ => Array.empty[BlockLocation]
  }

  // Given locations of all blocks of a single file, `blockLocations`, and an `(offset, length)`
  // pair that represents a segment of the same file, find out the block that contains the largest
  // fraction the segment, and returns location hosts of that block. If no such block can be found,
  // returns an empty array.
  private def getBlockHosts(
    blockLocations: Array[BlockLocation],
    offset: Long,
    length: Long): Array[String] = {
    val candidates = blockLocations.map {
      // The fragment starts from a position within this block
      case b if b.getOffset <= offset && offset < b.getOffset + b.getLength =>
        b.getHosts -> (b.getOffset + b.getLength - offset).min(length)

      // The fragment ends at a position within this block
      case b if offset <= b.getOffset && offset + length < b.getLength =>
        b.getHosts -> (offset + length - b.getOffset).min(length)

      // The fragment fully contains this block
      case b if offset <= b.getOffset && b.getOffset + b.getLength <= offset + length =>
        b.getHosts -> b.getLength

      // The fragment doesn't intersect with this block
      case b =>
        b.getHosts -> 0L
    }.filter { case (hosts, size) =>
      size > 0L
    }

    if (candidates.isEmpty) {
      Array.empty[String]
    } else {
      val (hosts, _) = candidates.maxBy { case (_, size) => size }
      hosts
    }
  }
}

object GpuDataset {
  private val logger = LogFactory.getLog(classOf[GpuDataset])
  private val PARQUET_MAGIC = "PAR1".getBytes(StandardCharsets.US_ASCII)

  private def getMapper[U: ClassTag](func: GpuColumnBatch => Iterator[U]):
      Iterator[GpuColumnBatch] => Iterator[U] = {
    batchIter: Iterator[GpuColumnBatch] => {
      if (batchIter.hasNext) {
        val batch = batchIter.next
        if (batchIter.hasNext) {
          throw new UnsupportedOperationException("Column batch iterator returned multiple batches")
        }
        func(batch)
      } else {
        Iterator.empty
      }
    }
  }

  private def maxDoubleMapper(columnIndex: Int): GpuColumnBatch => Iterator[Double] =
    (b: GpuColumnBatch) => {
      // call allocateGpuDevice to force assignment of GPU when in exclusive process mode
      // and pass that as the gpu_id, assumption is that if you are using CUDA_VISIBLE_DEVICES
      // it doesn't hurt to call allocateGpuDevice so just always do it.
      var gpuId = XGBoostSparkJNI.allocateGpuDevice()
      logger.debug("XGboost maxDoubleMapper get device: " + gpuId)

      val column = b.getColumnVector(columnIndex)
      val scalar = column.max()
      if (scalar.isValid) {
        Iterator.single(scalar.getDouble)
      } else {
        Iterator.empty
      }
    }

  private[xgboost4j] def columnBatchToRows(batch: GpuColumnBatch): Iterator[Row] = {
    val taskContext = TaskContext.get
    val iter = new Iterator[Row] with AutoCloseable {
      private val numRows = batch.getNumRows
      private val schema = batch.getSchema
      private val timeUnits =
        (0 until batch.getNumColumns).map(batch.getColumnVector(_).getTimeUnit)
      private val converter = new RowConverter(schema, timeUnits)
      private val rowSize = UnsafeRow.calculateBitSetWidthInBytes(batch.getNumColumns) +
        batch.getNumColumns * 8
      private var buffer: Long = _
      private var nextRow = 0
      private val row = new UnsafeRow(schema.length)

      override def hasNext: Boolean = nextRow < numRows

      override def next(): Row = {
        if (nextRow >= numRows) {
          throw new NoSuchElementException
        }
        if (buffer == 0) {
          initBuffer()
        }
        row.pointTo(null, buffer + rowSize * nextRow, rowSize)
        nextRow += 1
        converter.toExternalRow(row)
      }

      override def close(): Unit = {
        if (buffer != 0) {
          Platform.freeMemory(buffer)
          buffer = 0
        }
      }

      private def initBuffer(): Unit = {
        val nativeColumnPtrs = new Array[Long](batch.getNumColumns)
        for (i <- 0 until batch.getNumColumns) {
          nativeColumnPtrs(i) = batch.getColumn(i)
        }
        buffer = XGBoostSparkJNI.buildUnsafeRows(nativeColumnPtrs)
      }
    }

    taskContext.addTaskCompletionListener(_ => iter.close())
    iter
  }

  private def numericAsFloats(schema: StructType): StructType = {
    StructType(schema.fields.map {
      case StructField(name, nt: NumericType, nullable, metadata) =>
        StructField(name, FloatType, nullable, metadata)
      case other => other
    })
  }

  private def getPartFileReader(
        sparkSession: SparkSession,
        sourceType: String,
        schema: StructType,
        options: Map[String, String],
        castAllToFloats: Boolean): (Configuration, String, PartitionedFile) => Option[Table] = {
    sourceType match {
      case "csv" => getCsvPartFileReader(sparkSession, schema, options, castAllToFloats)
      case "parquet" => getParquetPartFileReader(sparkSession, schema, options, castAllToFloats)
      case "orc" => getOrcPartFileReader(sparkSession, schema, options, castAllToFloats)
      case _ => throw new UnsupportedOperationException(
        s"Unsupported source type: $sourceType")
    }
  }

  private def getCsvPartFileReader(
      sparkSession: SparkSession,
      inputSchema: StructType,
      options: Map[String, String],
      castAllToFloats: Boolean): (Configuration, String, PartitionedFile) => Option[Table] = {
    // Try to build CSV options on the driver here to validate and fail fast.
    // The other CSV option build below will occur on the executors.
    buildCsvOptions(options, true) // add virtual "isFirstSplit" for validation.
    val schema = if (castAllToFloats) {
      numericAsFloats(inputSchema)
    } else {
      inputSchema
    }
    (conf: Configuration, dumpPrefix: String, partFile: PartitionedFile) => {
      val (dataBuffer, dataSize) = readCsvPartFile(conf, dumpPrefix, partFile)
      try {

        if (dataSize == 0) {
          None
        } else {
          val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
          schema.foreach(f => csvSchemaBuilder.column(toDType(f.dataType), f.name))
          val table = Table.readCSV(csvSchemaBuilder.build(), buildCsvOptions(options,
            partFile.start == 0),
            dataBuffer, 0, dataSize)

          val numColumns = table.getNumberOfColumns
          if (schema.length != numColumns) {
            table.close()
            throw new QueryExecutionException(s"Expected ${schema.length} columns " +
              s"but only read ${table.getNumberOfColumns} from $partFile")
          }
          Some(table)
        }
      } finally {
        dataBuffer.close()
      }
    }
  }

  private def getParquetPartFileReader(
      sparkSession: SparkSession,
      schema: StructType,
      options: Map[String, String],
      castToFloat: Boolean): (Configuration, String, PartitionedFile) => Option[Table] = {
    // Try to build Parquet options on the driver here to validate and fail fast.
    // The other Parquet option build below will occur on the executors.
    buildParquetOptions(options, schema)

    (conf: Configuration, dumpPrefix: String, partFile: PartitionedFile) => {
      val (dataBuffer, dataSize) = readParquetPartFile(conf, partFile)
      try {
        if (dataSize == 0) {
          None
        } else {

          if (dumpPrefix != null) {
            dumpParquetData(dumpPrefix, dataBuffer, dataSize, conf)
          }
          val parquetOptions = buildParquetOptions(options, schema)
          var table = Table.readParquet(parquetOptions, dataBuffer, 0, dataSize)
          val numColumns = table.getNumberOfColumns

          if (schema.length != numColumns) {
            table.close()
            throw new QueryExecutionException(s"Expected ${schema.length} columns " +
              s"but read ${numColumns} from $partFile, partFile may be broken")
          }
          if (castToFloat) {
            val columns = new Array[ColumnVector](numColumns)
            try {
              for (i <- 0 until numColumns) {
                val c = table.getColumn(i)
                columns(i) = schema.fields(i).dataType match {
                  case nt: NumericType => c.asFloats()
                  case _ => c.incRefCount()
                }
              }
              var tmp = table
              table = new Table(columns: _*)
              tmp.close()
            } finally {
              columns.foreach(v => if (v != null) v.close())
            }
          }
          Some(table)
        }
      } finally {
        dataBuffer.close()
      }
    }
  }

  private def getOrcPartFileReader(
      sparkSession: SparkSession,
      schema: StructType,
      options: Map[String, String],
      castToFloat: Boolean
  ): (Configuration, String, PartitionedFile) => Option[Table] = {
    // Try to build ORC options on the driver here to validate and fail fast.
    // The other ORC option build below will occur on the executors.
    buildOrcOptions(options, schema)

    (configuration, dumpPrefix, partitionedFile) => {
      val (dataBuffer, dataSize) = readPartFileFully(configuration, partitionedFile)
      try {
        val table = Table.readORC(
            buildOrcOptions(options, schema),
            dataBuffer,
            0,
            dataSize)

        if (schema.length != table.getNumberOfColumns) {
          table.close()
          throw new IllegalStateException(
              s"Expected ${schema.length} columns " +
              s"but only read ${table.getNumberOfColumns} from ${partitionedFile}")
        }

        val columns = new Array[ColumnVector](schema.length)
        try {
          for (i <- 0 until schema.length) {
            val c = table.getColumn(i)
            columns(i) = schema(i).dataType match {
              case _: NumericType => if (castToFloat) c.asFloats() else c.incRefCount()
              // The GPU ORC reader always casts date and timestamp columns to DATE64.
              // See https://github.com/rapidsai/cudf/issues/2384
              // So casts it back for DATE32.
              case _: DateType => c.asDate32()
              case _ => c.incRefCount()
            }
          }
          Some(new Table(columns: _*))
        } finally {
          table.close()
          for (c <- columns if c != null) {
            c.close()
          }
        }
      } finally {
        dataBuffer.close()
      }
    }
  }

  private def buildCsvOptions(options: Map[String, String], isFirstSplit: Boolean): CSVOptions = {
    val builder = CSVOptions.builder()
    for ((k, v) <- options) {
      if (k == "header") {
        builder.hasHeader(getBool(k, v) && isFirstSplit)
      } else {
        val parseFunc = csvOptionParserMap.getOrElse(k, (_: CSVOptions.Builder, _: String) => {
          throw new UnsupportedOperationException(s"CSV option $k not supported")
        })
        parseFunc(builder, v)
      }
    }
    builder.build
  }

  private def buildParquetOptions(options: Map[String, String],
      schema: StructType): ParquetOptions = {
    // currently no Parquet read options are supported
    if (options.nonEmpty) {
      throw new UnsupportedOperationException("No Parquet read options are supported")
    }
    val builder = ParquetOptions.builder()
    builder.includeColumn(schema.map(_.name): _*)
    builder.build
  }

  private def buildOrcOptions(options: Map[String, String], schema: StructType): ORCOptions = {
    // currently no ORC read options are supported
    if (options.nonEmpty) {
      throw new UnsupportedOperationException("No ORC read options are supported")
    }
    ORCOptions
        .builder()
        .includeColumn(schema.map(_.name): _*)
        .build
  }

  private def readPartFileFully(conf: Configuration,
      partFile: PartitionedFile): (HostMemoryBuffer, Long) = {
    val rawPath = new Path(partFile.filePath)
    val fs = rawPath.getFileSystem(conf)
    val path = fs.makeQualified(rawPath)
    val fileSize = fs.getFileStatus(path).getLen
    var succeeded = false
    var hmb = HostMemoryBuffer.allocate(fileSize)
    var totalBytesRead: Long = 0L
    try {
      val buffer = new Array[Byte](1024 * 16)
      val codecFactory = new CompressionCodecFactory(conf)
      val codec = codecFactory.getCodec(path)
      val rawInput = fs.open(path)
      val in = if (codec != null) codec.createInputStream(rawInput) else rawInput
      try {
        var numBytes = in.read(buffer)
        while (numBytes >= 0) {
          if (totalBytesRead + numBytes > hmb.getLength) {
            hmb = growHostBuffer(hmb, totalBytesRead + numBytes)
          }
          hmb.setBytes(totalBytesRead, buffer, 0, numBytes)
          totalBytesRead += numBytes
          numBytes = in.read(buffer)
        }
      } finally {
        in.close()
      }
      succeeded = true
    } finally {
      if (!succeeded) {
        hmb.close()
      }
    }
    (hmb, totalBytesRead)
  }

  private def estimatedHostBufferSizeForCsv(conf: Configuration, partFile: PartitionedFile):
    Long = {
    val rawPath = new Path(partFile.filePath)
    val fs = rawPath.getFileSystem(conf)
    val path = fs.makeQualified(rawPath)
    val fileSize = fs.getFileStatus(path).getLen
    val codecFactory = new CompressionCodecFactory(conf)
    val codec = codecFactory.getCodec(path)
    if (codec != null) {
      // wild guess that compression is 2X or less
      partFile.length * 2
    } else if (partFile.start + partFile.length == fileSize){
      // last split doesn't need to read an additional record.
      // (this PartitionedFile is one complete file)
      partFile.length
    } else {
      // wild guess for extra space needed for the record after the split end offset
      partFile.length + 128 * 1024
    }
  }


  private def readCsvPartFile(conf: Configuration, dumpPrefix: String, partFile: PartitionedFile):
    (HostMemoryBuffer, Long) = {
    // use '\n' as line seperator.
    val seperator = Array('\n'.toByte)
    var succeeded = false
    var totalSize: Long = 0L
    var hmb = HostMemoryBuffer.allocate(estimatedHostBufferSizeForCsv(conf, partFile))
    try {
      val lineReader = new HadoopFileLinesReader(partFile, conf)
      try {
        while (lineReader.hasNext) {
          val line = lineReader.next()
          val lineSize = line.getLength
          val newTotal = totalSize + lineSize + seperator.length
          if (newTotal > hmb.getLength) {
            hmb = growHostBuffer(hmb, newTotal)
          }
          hmb.setBytes(totalSize, line.getBytes, 0, lineSize)
          hmb.setBytes(totalSize + lineSize, seperator, 0, seperator.length)
          totalSize = newTotal
        }
        succeeded = true
      } finally {
        lineReader.close()
      }
    } finally {
      if (!succeeded) {
        hmb.close()
      }
    }
    (hmb, totalSize)
  }

  /**
    * Grows a host buffer, returning a new buffer and closing the original
    * after copying the data into the new buffer.
    * @param original the original host memory buffer
    */
  private def growHostBuffer(original: HostMemoryBuffer, needed: Long): HostMemoryBuffer = {
    val newSize = Math.max(original.getLength * 2, needed)
    val result = HostMemoryBuffer.allocate(newSize)
    try {
      result.copyFromHostBuffer(0, original, 0, original.getLength)
      original.close()
    } catch {
      case e: Throwable =>
        result.close()
        throw e
    }
    result
  }


  private def readParquetPartFile(conf: Configuration, partFile: PartitionedFile):
    (HostMemoryBuffer, Long) = {

    val filePath = new Path(new URI(partFile.filePath))
    val footer = ParquetFileReader.readFooter(conf, filePath,
      ParquetMetadataConverter.range(partFile.start, partFile.start + partFile.length))
    val fileSchema = footer.getFileMetaData.getSchema
    val blocks = footer.getBlocks.asScala

    if (blocks.length == 0) {
      (HostMemoryBuffer.allocate(0), 0)
    } else {
      val in = filePath.getFileSystem(conf).open(filePath)
      try {
        var succeeded = false
        val hmb = HostMemoryBuffer.allocate(calculateParquetOutputSize(blocks, fileSchema))
        try {
          val out = new HostMemoryBufferOutputStream(hmb)
          out.write(GpuDataset.PARQUET_MAGIC)
          val outputBlocks = copyBlocksData(in, out, blocks, partFile)
          val footerPos = out.getPos
          writeFooter(out, outputBlocks, fileSchema)
          BytesUtils.writeIntLittleEndian(out, (out.getPos - footerPos).toInt)
          out.write(GpuDataset.PARQUET_MAGIC)
          succeeded = true
          (hmb, out.getPos)
        } finally {
          if (!succeeded) {
            hmb.close()
          }
        }
      } finally {
        in.close()
      }
    }
  }


  private def calculateParquetOutputSize(
      blocks: Seq[BlockMetaData],
      parquetSchema: MessageType): Long = {
    // parquet foramt requirements
    var size: Long = 4 + 4 + 4
    size += blocks.flatMap(b => b.getColumns.asScala.map(_.getTotalSize)).sum
    // Calculate size of the footer metadata.
    // This uses the column metadata from the original file, but that should
    // always be at least as big as the updated metadata in the output.
    val out = new CountingOutputStream(new NullOutputStream)
    writeFooter(out, blocks, parquetSchema)
    size + out.getByteCount
  }

  private[spark] def clipBlocks(columnPaths: Seq[ColumnPath], blocks: Seq[BlockMetaData]):
    Seq[BlockMetaData] = {
    val pathSet = columnPaths.toSet
    blocks.map(oldBlock => {
      // noinspection ScalaDeprecation
      val newColumns = oldBlock.getColumns.asScala.filter(c => pathSet.contains(c.getPath))
      newParquetBlock(oldBlock.getRowCount, newColumns)
    })
  }

  private def copyBlocksData(
       in: FSDataInputStream,
       out: HostMemoryBufferOutputStream,
       blocks: Seq[BlockMetaData],
       split: PartitionedFile): Seq[BlockMetaData] = {
    var totalRows: Long = 0
    val copyBuffer = new Array[Byte](128 * 1024)
    val outputBlocks = new ArrayBuffer[BlockMetaData](blocks.length)
    for (block <- blocks) {
      totalRows += block.getRowCount
      if (totalRows > Integer.MAX_VALUE) {
        throw new UnsupportedOperationException(s"Too many rows in split $split")
      }
      val columns = block.getColumns.asScala
      val outputColumns = new ArrayBuffer[ColumnChunkMetaData](columns.length)
      for (column <- columns) {
        // update column metadata to reflect new position in the output file
        val offsetAdjustment = out.getPos - column.getStartingPos
        val newDictOffset = if (column.getDictionaryPageOffset > 0) {
          column.getDictionaryPageOffset + offsetAdjustment
        } else {
          0
        }
        // noinspection ScalaDeprecation
        outputColumns += ColumnChunkMetaData.get(
          column.getPath,
          column.getType,
          column.getCodec,
          column.getEncodingStats,
          column.getEncodings,
          column.getStatistics,
          column.getFirstDataPageOffset + offsetAdjustment,
          newDictOffset,
          column.getValueCount,
          column.getTotalSize,
          column.getTotalUncompressedSize)
        copyColumnData(column, in, out, copyBuffer)
      }
      outputBlocks += newParquetBlock(block.getRowCount, outputColumns)
    }
    outputBlocks
  }


  private def newParquetBlock(
      rowCount: Long,
      columns: Seq[ColumnChunkMetaData]): BlockMetaData = {
    val block = new BlockMetaData
    block.setRowCount(rowCount)

    var totalSize: Long = 0
    columns.foreach { column =>
      block.addColumn(column)
      totalSize += column.getTotalUncompressedSize
    }
    block.setTotalByteSize(totalSize)
    block
  }

  private def copyColumnData(
       column: ColumnChunkMetaData,
       in: FSDataInputStream,
       out: OutputStream,
       copyBuffer: Array[Byte]): Unit = {
    if (in.getPos != column.getStartingPos) {
      in.seek(column.getStartingPos)
    }
    var bytesLeft = column.getTotalSize
    while (bytesLeft > 0) {
      // downcast is safe because copyBuffer.length is an int
      val readLength = Math.min(bytesLeft, copyBuffer.length).toInt
      in.readFully(copyBuffer, 0, readLength)
      out.write(copyBuffer, 0, readLength)
      bytesLeft -= readLength
    }
  }



  private def writeFooter(
      out: OutputStream,
      blocks: Seq[BlockMetaData],
      parquetSchema: MessageType): Unit = {
    val PARQUET_CREATOR = "RAPIDS GpuDataset"
    val PARQUET_VERSION = 1
    val fileMeta = new FileMetaData(parquetSchema, Collections.emptyMap[String, String],
      PARQUET_CREATOR)
    val metadataConverter = new ParquetMetadataConverter
    val footer = new ParquetMetadata(fileMeta, blocks.asJava)
    val meta = metadataConverter.toParquetMetadata(PARQUET_VERSION, footer)
    org.apache.parquet.format.Util.writeFileMetaData(meta, out)
  }

  private class HostMemoryBufferOutputStream(buffer: HostMemoryBuffer) extends OutputStream {
    private var pos: Long = 0

    override def write(i: Int): Unit = {
      buffer.setByte(pos, i.toByte)
      pos += 1
    }

    override def write(bytes: Array[Byte]): Unit = {
      buffer.setBytes(pos, bytes, 0, bytes.length)
      pos += bytes.length
    }

    override def write(bytes: Array[Byte], offset: Int, len: Int): Unit = {
      buffer.setBytes(pos, bytes, offset, len)
      pos += len
    }

    def getPos: Long = pos
  }


  private def dumpParquetData(
      dumpPathPrefix: String,
      hmb: HostMemoryBuffer,
      dataLength: Long,
      conf: Configuration): Unit = {
    val (out, path) = FileUtils.createTempFile(conf, dumpPathPrefix, ".parquet")
    try {
      logger.info(s"Writing Parquet split data to $path")
      val buffer = new Array[Byte](128 * 1024)
      var pos = 0
      while (pos < dataLength) {
        // downcast is safe because buffer.length is an int
        val readLength = Math.min(dataLength - pos, buffer.length).toInt
        hmb.getBytes(buffer, 0, pos, readLength)
        out.write(buffer, 0, readLength)
        pos += readLength
      }
    } finally {
      out.close()
    }
  }



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

  /**
   * Helper method that converts string representation of a character to actual character.
   * It handles some Java escaped strings and throws exception if given string is longer than one
   * character.
   */
  @throws[IllegalArgumentException]
  private def toChar(str: String): Char = {
    (str: Seq[Char]) match {
      case Seq() => throw new IllegalArgumentException("Delimiter cannot be empty string")
      case Seq('\\') => throw new IllegalArgumentException("Single backslash is prohibited." +
        " It has special meaning as beginning of an escape sequence." +
        " To get the backslash character, pass a string with two backslashes as the delimiter.")
      case Seq(c) => c
      case Seq('\\', 't') => '\t'
      case Seq('\\', 'r') => '\r'
      case Seq('\\', 'b') => '\b'
      case Seq('\\', 'f') => '\f'
      // In case user changes quote char and uses \" as delimiter in options
      case Seq('\\', '\"') => '\"'
      case Seq('\\', '\'') => '\''
      case Seq('\\', '\\') => '\\'
      case _ if str == """\u0000""" => '\u0000'
      case Seq('\\', _) =>
        throw new IllegalArgumentException(s"Unsupported special character for delimiter: $str")
      case _ =>
        throw new IllegalArgumentException(s"Delimiter cannot be more than one character: $str")
    }
  }

  private def getBool(paramName: String, paramVal: String): Boolean = {
    val lowerParamVal = paramVal.toLowerCase(Locale.ROOT)
    if (lowerParamVal == "true") {
      true
    } else if (lowerParamVal == "false") {
      false
    } else {
      throw new Exception(s"$paramName flag can be true or false")
    }
  }

  private def parseCSVCommentOption(b: CSVOptions.Builder, v: String): Unit = {
    b.withComment(toChar(v))
  }

  private def parseCSVHeaderOption(b: CSVOptions.Builder, v: String): Unit = {
    if (getBool("header", v)) {
      b.hasHeader
    }
  }

  private def parseCSVNullValueOption(b: CSVOptions.Builder, v: String): Unit = {
    b.withNullValue(v)
  }

  private def parseCSVQuoteOption(b: CSVOptions.Builder, v: String): Unit = {
    b.withQuote(toChar(v))
  }

  private def parseCSVSepOption(b: CSVOptions.Builder, v: String): Unit = {
    b.withDelim(toChar(v))
  }

  private val csvOptionParserMap: Map[String, (CSVOptions.Builder, String) => Unit] = Map(
    "comment" -> parseCSVCommentOption,
    "header" -> parseCSVHeaderOption,
    "nullValue" -> parseCSVNullValueOption,
    "quote" -> parseCSVQuoteOption,
    "sep" -> parseCSVSepOption
  )

  private def createFilePartition(
      size: Int,
      files: Seq[PartitionedFile]): FilePartition = {
    // Apache Spark:
    // case class FilePartition(index: Int, files: Array[PartitionedFile])
    // We have see this overloaded as:
    // case class FilePartition(index: Int, files: Seq[PartitionedFile],
    //   preferredHosts: Seq[String])
    // We skip setting the preferredHosts for overloaded version
    try {
      FilePartition(size, files)
    } catch {
      case e: NoSuchMethodError =>
        logger.debug("FilePartition, normal Apache Spark version failed")
        // assume this is the one overloaded class we know about
        val fpClass = org.apache.spark.sql.execution.datasources.FilePartition.getClass.
          getMethod("apply", classOf[Int], classOf[Seq[PartitionedFile]], classOf[Seq[String]])
        fpClass.invoke(org.apache.spark.sql.execution.datasources.FilePartition,
          new java.lang.Integer(size), files, Seq.empty[String]).asInstanceOf[FilePartition]
    }
  }

  private class PartBucket(initFile: PartitionedFile) {
    private val files: ArrayBuffer[PartitionedFile] = ArrayBuffer(initFile)
    private var size = initFile.length

    def getSize: Long = size

    def addFile(file: PartitionedFile): Unit = {
      files += file
      size += file.length
    }

    def toFilePartition(index: Int): FilePartition = createFilePartition(index, files)
  }
}
