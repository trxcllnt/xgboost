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

import java.sql.Date

import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.FunSuite

trait SparkQueryCompareTestSuite extends FunSuite {
   def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

  def gpuFrameFromParquet(filename: String): SparkSession => GpuDataset = {
    s: SparkSession => new GpuDataReader(s).parquet(filename.toString)
  }

  def withSparkSession[U](ss: SparkSession, appName: String,
      conf: SparkConf, f: SparkSession => U): U = {
    try {
      f(ss)
    } finally {
      ss.stop()
      SparkSession.clearActiveSession()
      SparkSession.clearDefaultSession()
    }
  }

  def runOnCpuAndGpu(
      cpuDf: DataFrame,
      gpuDf: GpuDataset,
      conf: SparkConf = new SparkConf(),
      repart: Integer = 1): (Array[Row], Array[Row]) = {
    val fromCpu = {
      var data = cpuDf
      if (repart > 0) {
        // repartition the data so it is turned into a projection,
        // not folded into the table scan exec
        data = data.repartition(repart)
      }
      data.collect()
    }

    val fromGpu = {
      var data = gpuDf
      if (repart > 0) {
        // repartition the data so it is turned into a projection,
        // not folded into the table scan exec
        data = data.repartition(repart)
      }
      val rdd = data.buildRDD.mapPartitions(GpuDataset.columnBatchToRows)
      rdd.collect()
    }
    (fromCpu, fromGpu)
  }

  // we guarantee that the types will be the same
  private def seqLt(a: Seq[Any], b: Seq[Any]): Boolean = {
    if (a.length < b.length) {
      return true
    }
    // lengths are the same
    for (i <- a.indices) {
      val v1 = a(i)
      val v2 = b(i)
      if (v1 != v2) {
        // null is always < anything but null
        if (v1 == null) {
          return true
        }

        if (v2 == null) {
          return false
        }

        return if ((v1, v2) match {
          case (i1: Int, i2: Int) => i1 < i2
          case (i1: Long, i2: Long) => i1 < i2
          case (i1: Float, i2: Float) => i1 < i2
          case (i1: Date, i2: Date) => i1.before(i2)
          case (i1: Double, i2: Double) => i1 < i2
          case (i1: Short, i2: Short) => i1 < i2
          case (o1, o2) => throw new UnsupportedOperationException(
            o1.getClass + " is not supported yet")
        }) {
          true
        } else {
          false
        }
      }
    }
    false
  }

  private def compare(obj1: Any, obj2: Any, maxFloatDiff: Double = 0.0):
  Boolean = (obj1, obj2) match {
    case (null, null) => true
    case (null, _) => false
    case (_, null) => false
    case (a: Array[_], b: Array[_]) =>
      a.length == b.length && a.zip(b).forall { case (l, r) => compare(l, r, maxFloatDiff)}
    case (a: Map[_, _], b: Map[_, _]) =>
      a.size == b.size && a.keys.forall { aKey =>
        b.keys.find(bKey => compare(aKey, bKey)).exists(
          bKey => compare(a(aKey), b(bKey), maxFloatDiff))
      }
    case (a: Iterable[_], b: Iterable[_]) =>
      a.size == b.size && a.zip(b).forall { case (l, r) => compare(l, r, maxFloatDiff)}
    case (a: Product, b: Product) =>
      compare(a.productIterator.toSeq, b.productIterator.toSeq, maxFloatDiff)
    case (a: Row, b: Row) =>
      compare(a.toSeq, b.toSeq, maxFloatDiff)
    // 0.0 == -0.0, turn float/double to bits before comparison, to distinguish 0.0 and -0.0.
    case (a: Double, b: Double) if maxFloatDiff <= 0 =>
      java.lang.Double.doubleToRawLongBits(a) == java.lang.Double.doubleToRawLongBits(b)
    case (a: Double, b: Double) if maxFloatDiff > 0 =>
      val ret = (Math.abs(a - b) <= maxFloatDiff)
      if (!ret) {
        System.err.println(
          s"\n\nABS(${a} - ${b}) == ${Math.abs(a - b)} is not <= ${maxFloatDiff} (double)")
      }
      ret
    case (a: Float, b: Float) if maxFloatDiff <= 0 =>
      java.lang.Float.floatToRawIntBits(a) == java.lang.Float.floatToRawIntBits(b)
    case (a: Float, b: Float) if maxFloatDiff > 0 =>
      val ret = (Math.abs(a - b) <= maxFloatDiff)
      if (!ret) {
        System.err.println(
          s"\n\nABS(${a} - ${b}) == ${Math.abs(a - b)} is not <= ${maxFloatDiff} (float)")
      }
      ret
    case (a, b) => a == b
  }

  def compareResults(
      sort: Boolean,
      maxFloatDiff: Double,
      fromCpu: Array[Row],
      fromGpu: Array[Row]): Unit = {
    val relaxedFloatDisclaimer = if (maxFloatDiff > 0) {
      "(relaxed float comparison)"
    } else {
      ""
    }
    if (sort) {
      val cpu = fromCpu.map(_.toSeq).sortWith(seqLt)
      val gpu = fromGpu.map(_.toSeq).sortWith(seqLt)
      if (!compare(cpu, gpu, maxFloatDiff)) {
        fail(
          s"""
             |Running on the GPU and on the CPU did not match $relaxedFloatDisclaimer
             |CPU: ${cpu.seq}

             |GPU: ${gpu.seq}
         """.
            stripMargin)
      }
    } else {
      if (!compare(fromCpu, fromGpu, maxFloatDiff)) {
        fail(
          s"""
             |Running on the GPU and on the CPU did not match $relaxedFloatDisclaimer
             |CPU: ${fromCpu.toSeq}

             |GPU: ${fromGpu.toSeq}
         """.
            stripMargin)
      }
    }
  }
}
