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

import ml.dmlc.xgboost4j.scala.spark.nvidia.NVDataReader
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types._
import org.scalatest.FunSuite

class XGBoostRegressorNVSuite extends FunSuite with PerTest {

  private val NVRegressor = new XGBoostRegressor(Map(
    "silent" -> 1,
    "eta" -> 0.1f,
    "max_depth" -> 3,
    "objective" -> "reg:squarederror",
    "num_round" -> 50,
    "num_workers" -> 1,
    "timeout_request_workers" -> 60000L))

  test("test XGBoost-Spark XGBoostRegressor setFeaturesCols") {
    val gdfCols = Seq("gdfCol1", "gdfCol2")
    NVRegressor.setFeaturesCols(gdfCols)
    assert(NVRegressor.getFeaturesCols.contains("gdfCol1"))
    assert(NVRegressor.getFeaturesCols.contains("gdfCol2"))
    assert(NVRegressor.getFeaturesCols.length == 2)
  }

  test("test XGBoost-Spark XGBoostRegressor the overloaded 'fit' should work with NVDataset") {
    val csvSchema = new StructType()
      .add("b", DoubleType)
      .add("c", DoubleType)
      .add("d", DoubleType)
      .add("e", IntegerType)
    // group now is not supported
    val trainDataAsNVDS = new NVDataReader(ss).schema(csvSchema).csv(getPath("norank.train.csv"))
    NVRegressor.setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
    NVRegressor.setLabelCol("e")
    val model = NVRegressor.fit(trainDataAsNVDS)
    val ret = model.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < ret && ret < 20)
  }
}
