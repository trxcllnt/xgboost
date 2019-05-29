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

import java.io.ByteArrayOutputStream
import java.util.Locale

import ai.rapids.cudf.{CSVOptions, DType, ParquetOptions, Table}
import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import ml.dmlc.xgboost4j.java.XGBoostSparkJNI
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{BlockLocation, FileStatus, FileSystem, LocatedFileStatus, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.datasources.{FilePartition, HadoopFsRelation, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.types.{BooleanType, ByteType, DataType, DateType, DoubleType, FloatType, IntegerType, LongType, ShortType, StructType, TimestampType}
import org.apache.spark.{SerializableWritable, TaskContext}
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.unsafe.Platform

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class NVDataset(fsRelation: HadoopFsRelation,
    sourceType: String,
    sourceOptions: Map[String, String],
    specifiedPartitions: Option[Seq[FilePartition]] = None) {

  private val logger = LogFactory.getLog(classOf[NVDataset])

  /** Returns the schema of the data. */
  def schema: StructType = fsRelation.schema

  /**
    *  Return an [[RDD]] of column batches.
    *  NOTE: NVColumnBatch is NOT serializable, so one has to be very careful in
    *        the types of operations performed on this RDD!
    */
  private[xgboost4j] def buildRDD: RDD[NVColumnBatch] = {
    val partitionReader = NVDataset.getPartFileReader(fsRelation.sparkSession, sourceType,
      schema, sourceOptions)
    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val serializableConf = new SerializableWritable[Configuration](hadoopConf)
    val broadcastedConf = sparkSession.sparkContext.broadcast(serializableConf)
    new NVDatasetRDD(fsRelation.sparkSession, broadcastedConf, partitionReader, partitions, schema)
  }

  /**
    * Return an [[RDD]] by applying a function expecting RAPIDS cuDF column pointers
    * to each partition of this data.
    */
  private[xgboost4j] def mapColumnarSingleBatchPerPartition[U: ClassTag](
      func: NVColumnBatch => Iterator[U]): RDD[U] = {
    buildRDD.mapPartitions(NVDataset.getMapper(func))
  }

  /**
    * Zip the data partitions with another RDD and return a new RDD by
    * applying a function to the zipped partitions. Assumes that this data and the RDD have the
    * *same number of partitions*, but does *not* require them to have the same number
    * of elements in each partition.
    */
  private[xgboost4j] def zipPartitionsAsRows[B: ClassTag, V: ClassTag]
      (other: RDD[B], preservesPartitioning: Boolean)
      (f: (Iterator[Row], Iterator[B]) => Iterator[V]): RDD[V] = {
    val rdd = mapColumnarSingleBatchPerPartition(NVDataset.columnBatchToRows)
    rdd.zipPartitions(other, preservesPartitioning)(f)
  }

  /** Return a new NVDataset that has exactly numPartitions partitions. */
  def repartition(numPartitions: Int): NVDataset = {
    if (numPartitions == partitions.length) {
      return new NVDataset(fsRelation, sourceType, sourceOptions, Some(partitions))
    }

    // build a list of all input files sorted from largest to smallest
    val files = partitions.flatMap(_.files).sortBy(_.length)(Ordering[Long].reverse)
    // Currently do not support splitting files
    if (files.length < numPartitions) {
      throw new UnsupportedOperationException("Cannot create more partitions than input files")
    }

    // Seed the partition buckets with one of the largest files then iterate
    // through the rest of the files, adding each to the smallest bucket
    val buckets = files.take(numPartitions).map(new NVDataset.PartBucket(_))
    def bucketOrder(b: NVDataset.PartBucket) = -b.getSize
    val queue = mutable.PriorityQueue(buckets: _*)(Ordering.by(bucketOrder))
    for (file <- files.drop(numPartitions)) {
      val bucket = queue.dequeue()
      bucket.addFile(file)
      queue.enqueue(bucket)
    }

    val newPartitions = buckets.zipWithIndex.map{case (b, i) => b.toFilePartition(i)}
    new NVDataset(fsRelation, sourceType, sourceOptions, Some(newPartitions))
  }

  /** The SparkSession that created this NVDataset. */
  def sparkSession: SparkSession = fsRelation.sparkSession

  private[nvidia] lazy val partitions: Seq[FilePartition] = specifiedPartitions.getOrElse{
    val selectedPartitions = fsRelation.location.listFiles(Nil, Nil)
    val openCostInBytes = fsRelation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val maxSplitBytes = computeMaxSplitBytes(fsRelation.sparkSession, selectedPartitions)
    logger.info(s"Planning scan with bin packing, max size: $maxSplitBytes bytes, " +
      s"open cost is considered as scanning $openCostInBytes bytes.")

    val splits = selectedPartitions.flatMap { partition =>
      partition.files.filter(_.getLen > 0).flatMap { file =>
        // getPath() is very expensive so we only want to call it once in this block:
        val filePath = file.getPath
        val isSplitable = fsRelation.fileFormat.isSplitable(
          fsRelation.sparkSession, fsRelation.options, filePath)
        splitFile(
          sparkSession = fsRelation.sparkSession,
          file = file,
          filePath = filePath,
          isSplitable = isSplitable,
          maxSplitBytes = maxSplitBytes,
          partitionValues = partition.values
        )
      }
    }.toArray.sortBy(_.length)(implicitly[Ordering[Long]].reverse)

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
        val newPartition = FilePartition(partitions.size, currentFiles.toArray.toSeq)
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
      selectedPartitions: Seq[PartitionDirectory]): Long = {
    val defaultMaxSplitBytes = sparkSession.sessionState.conf.filesMaxPartitionBytes
    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    val defaultParallelism = sparkSession.sparkContext.defaultParallelism
    val totalBytes = selectedPartitions.flatMap(_.files.map(_.getLen + openCostInBytes)).sum
    val bytesPerCore = totalBytes / defaultParallelism

    Math.min(defaultMaxSplitBytes, Math.max(openCostInBytes, bytesPerCore))
  }

  private def splitFile(
      sparkSession: SparkSession,
      file: FileStatus,
      filePath: Path,
      isSplitable: Boolean,
      maxSplitBytes: Long,
      partitionValues: InternalRow): Seq[PartitionedFile] = {
    // Currently there is no support for splitting a single file.
    if (false) {  // if (isSplitable) {
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

object NVDataset {
  private val logger = LogFactory.getLog(classOf[NVDataset])

  private def getMapper[U: ClassTag](func: NVColumnBatch => Iterator[U]):
      Iterator[NVColumnBatch] => Iterator[U] = {
    batchIter: Iterator[NVColumnBatch] => {
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

  private def columnBatchToRows(batch: NVColumnBatch): Iterator[Row] = {
    val taskContext = TaskContext.get
    val iter = new Iterator[Row] with AutoCloseable {
      private val numRows = batch.getNumRows
      private val schema = batch.getSchema
      private val converter = new RowConverter(schema)
      private val rowSize = UnsafeRow.calculateBitSetWidthInBytes(schema.length) + schema.length * 8
      private var buffer: Long = _
      private var nextRow = 0
      private val row = new UnsafeRow(batch.getNumColumns)

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

  private def getPartFileReader(
        sparkSession: SparkSession,
        sourceType: String,
        schema: StructType,
        options: Map[String, String]): (Configuration, PartitionedFile) => Table = {
    sourceType match {
      case "csv" => getCsvPartFileReader(sparkSession, schema, options)
      case "parquet" => getParquetPartFileReader(sparkSession, schema, options)
      case _ => throw new UnsupportedOperationException(
        s"Unsupported source type: $sourceType")
    }
  }

  private def getCsvPartFileReader(
      sparkSession: SparkSession,
      schema: StructType,
      options: Map[String, String]): (Configuration, PartitionedFile) => Table = {
    // Try to build CSV options on the driver here to validate and fail fast.
    // The other CSV option build below will occur on the executors.
    buildCsvOptions(options)

    (conf: Configuration, partFile: PartitionedFile) => {
      val partFileData = readPartFileFully(partFile, conf)
      val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
      schema.foreach(f => csvSchemaBuilder.column(toDType(f.dataType), f.name))
      val table = Table.readCSV(csvSchemaBuilder.build(), buildCsvOptions(options), partFileData)
      val numColumns = table.getNumberOfColumns
      if (schema.length != numColumns) {
        table.close()
        throw new QueryExecutionException(s"Expected ${schema.length} columns " +
          s"but only read ${table.getNumberOfColumns} from $partFile")
      }
      table
    }
  }

  private def getParquetPartFileReader(
      sparkSession: SparkSession,
      schema: StructType,
      options: Map[String, String]): (Configuration, PartitionedFile) => Table = {
    // Try to build Parquet options on the driver here to validate and fail fast.
    // The other Parquet option build below will occur on the executors.
    buildParquetOptions(options, schema)

    (conf: Configuration, partFile: PartitionedFile) => {
      val partFileData = readPartFileFully(partFile, conf)
      val parquetOptions = buildParquetOptions(options, schema)
      val table = Table.readParquet(parquetOptions, partFileData)
      val numColumns = table.getNumberOfColumns
      if (schema.length != numColumns) {
        table.close()
        throw new QueryExecutionException(s"Expected ${schema.length} columns " +
            s"but only read ${table.getNumberOfColumns} from $partFile")
      }
      table
    }
  }

  private def buildCsvOptions(options: Map[String, String]): CSVOptions = {
    val builder = CSVOptions.builder()
    for ((k, v) <- options) {
      val parseFunc = csvOptionParserMap.getOrElse(k, (_: CSVOptions.Builder, _: String) => {
        throw new UnsupportedOperationException(s"CSV option $k not supported")
      })
      parseFunc(builder, v)
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

  private def readPartFileFully(partFile: PartitionedFile,
      hadoopConf: Configuration): Array[Byte] = {
    val lfs = FileSystem.getLocal(hadoopConf)
    val path = lfs.makeQualified(new Path(partFile.filePath))
    val fs = path.getFileSystem(hadoopConf)
    val fileSize = fs.getFileStatus(path).getLen
    if (fileSize > Integer.MAX_VALUE) {
      throw new UnsupportedOperationException(s"File partition at $path is" +
        s" too big to buffer directly ($fileSize bytes)")
    }

    val baos = new ByteArrayOutputStream(fileSize.toInt)
    val buffer = new Array[Byte](1024 * 16)
    val in = fs.open(path)
    try {
      var numBytes = in.read(buffer)
      while (numBytes >= 0) {
        baos.write(buffer, 0, numBytes)
        numBytes = in.read(buffer)
      }
      baos.toByteArray
    } finally {
      in.close()
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

  private class PartBucket(initFile: PartitionedFile) {
    private val files: ArrayBuffer[PartitionedFile] = ArrayBuffer(initFile)
    private var size = initFile.length

    def getSize: Long = size

    def addFile(file: PartitionedFile): Unit = {
      files += file
      size += file.length
    }

    def toFilePartition(index: Int): FilePartition = FilePartition(index, files)
  }
}
