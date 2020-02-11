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

import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.rapids.ColumnBatchToRow
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.types.{FloatType, IntegerType}

object DataUtils extends Serializable {
  private[spark] implicit class XGBLabeledPointFeatures(
      val labeledPoint: XGBLabeledPoint
  ) extends AnyVal {
    /** Converts the point to [[MLLabeledPoint]]. */
    private[spark] def asML: MLLabeledPoint = {
      MLLabeledPoint(labeledPoint.label, labeledPoint.features)
    }

    /**
     * Returns feature of the point as [[org.apache.spark.ml.linalg.Vector]].
     *
     * If the point is sparse, the dimensionality of the resulting sparse
     * vector would be [[Int.MaxValue]]. This is the only safe value, since
     * XGBoost does not store the dimensionality explicitly.
     */
    def features: Vector = if (labeledPoint.indices == null) {
      Vectors.dense(labeledPoint.values.map(_.toDouble))
    } else {
      Vectors.sparse(Int.MaxValue, labeledPoint.indices, labeledPoint.values.map(_.toDouble))
    }
  }

  private[spark] implicit class MLLabeledPointToXGBLabeledPoint(
      val labeledPoint: MLLabeledPoint
  ) extends AnyVal {
    /** Converts an [[MLLabeledPoint]] to an [[XGBLabeledPoint]]. */
    def asXGB: XGBLabeledPoint = {
      labeledPoint.features.asXGB.copy(label = labeledPoint.label.toFloat)
    }
  }

  private[spark] implicit class MLVectorToXGBLabeledPoint(val v: Vector) extends AnyVal {
    /**
     * Converts a [[Vector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    def asXGB: XGBLabeledPoint = v match {
      case v: DenseVector =>
        XGBLabeledPoint(0.0f, null, v.values.map(_.toFloat))
      case v: SparseVector =>
        XGBLabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
    }
  }

  private[spark] def convertDataFrameToXGBLabeledPointRDDs(
      labelCol: Column,
      featuresCol: Column,
      weight: Column,
      baseMargin: Column,
      group: Option[Column],
      dataFrames: DataFrame*): Array[RDD[XGBLabeledPoint]] = {
    val selectedColumns = group.map(groupCol => Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      groupCol.cast(IntegerType),
      baseMargin.cast(FloatType))).getOrElse(Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      baseMargin.cast(FloatType)))
    dataFrames.toArray.map {
      df => df.select(selectedColumns: _*).rdd.map {
        case Row(label: Float, features: Vector, weight: Float, group: Int, baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          XGBLabeledPoint(label, indices, values, weight, group, baseMargin)
        case Row(label: Float, features: Vector, weight: Float, baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          XGBLabeledPoint(label, indices, values, weight, baseMargin = baseMargin)
      }
    }
  }

  private[spark] def buildDMatrixIncrementally(gpuId: Int, missing: Float,
      indices: Seq[Array[Int]], iter: Iterator[GpuColumnBatch],
      colNameToBuild: Option[String] = None): (DMatrix, ColumnBatchToRow) = {
    var dm: DMatrix = null
    var isFirstBatch = true
    val columnBatchToRow: ColumnBatchToRow = new ColumnBatchToRow

    while (iter.hasNext) {
      val columnBatch = iter.next()
      columnBatch.samplingTable()
      val gdfColsHandles = indices.map(_.map(columnBatch.getColumn))

      if (isFirstBatch) {
        isFirstBatch = false
        dm = new DMatrix(gdfColsHandles(0), gpuId, missing)
      } else {
        dm.appendCUDF(gdfColsHandles(0))
      }

      columnBatchToRow.appendColumnBatch(columnBatch, colNameToBuild)
      columnBatch.samplingClose()
    }
    if (dm == null) {
      // here we allow empty iter
      // throw new RuntimeException("Can't build Dmatrix from CUDF")
    }
    (dm, columnBatchToRow)
  }

}
