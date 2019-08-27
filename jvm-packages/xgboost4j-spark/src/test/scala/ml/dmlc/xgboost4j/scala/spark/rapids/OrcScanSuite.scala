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

import java.sql.{Date, Timestamp}

import ai.rapids.cudf.Cuda
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.SparkConf
import org.scalatest.FunSuite

class OrcScanSuite extends FunSuite with PerTest with SparkQueryCompareTestSuite {
  private lazy val SAMPLE_ORC_PATH = "/sample.orc"

  test("Test Orc parsing should be same for cpu and gpu ") {
    assume(Cuda.isEnvCompatibleForTesting)

    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.orc(getTestDataPath("/file-splits.orc")),
      new GpuDataReader(ss)
        .option("asFloats", false)
        .orc(getTestDataPath("/file-splits.orc")),
      conf = new SparkConf(),
      repart = 1)
    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }

  test("Test Orc Chunk") {
    assume(Cuda.isEnvCompatibleForTesting)
    val (fromCpu, fromGpu) = runOnCpuAndGpu(
      ss.read.orc(getTestDataPath("/file-splits.orc")),
      new GpuDataReader(ss)
        .option("maxRowsPerChunk", 2048)
        .option("asFloats", false)
        .orc(getTestDataPath("/file-splits.orc")),
      conf = new SparkConf(),
      repart = 1)

    compareResults(false, 0.000000001, fromCpu, fromGpu)
  }


  test("ORC parsing with raw types") {
    assume(Cuda.isEnvCompatibleForTesting)
    val data = new GpuDataReader(ss)
      .option("asFloats", false)
      .orc(getTestDataPath(SAMPLE_ORC_PATH))
      .buildRDD
      .mapPartitions(GpuDataset.columnBatchToRows)
      .collect

    assertResult(2) { data.length }

    val firstRow = data.head
    val secondRow = data.last
    assertResult(false) { firstRow.getBoolean(0) }
    assertResult(2) {  firstRow.getInt(1) }
    assertResult(6.25) { firstRow.getDouble(3) }
    assertResult(new Timestamp(1564999726000L)) { firstRow.getTimestamp(5) }

    assertResult(true) { secondRow.getBoolean(0) }
    assertResult(5.5) { secondRow.getFloat(2) }
    assertResult(Date.valueOf("2019-08-06")) { secondRow.getDate(4) }
  }

}
