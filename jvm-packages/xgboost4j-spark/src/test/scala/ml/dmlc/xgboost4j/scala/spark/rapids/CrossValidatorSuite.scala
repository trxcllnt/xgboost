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

import ml.dmlc.xgboost4j.scala.spark.{PerTest, XGBoostClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.scalatest.FunSuite

class CrossValidatorSuite extends FunSuite with PerTest {
  test("CrossValidator") {
    val labelCol = "classIndex"
    val featureCols: Seq[String] =
      Seq("sepal_length", "sepal_width", "petal_length", "petal_width")
    val schema = new StructType(Array(
      StructField("sepal_length", FloatType),
      StructField("sepal_width", FloatType),
      StructField("petal_length", FloatType),
      StructField("petal_width", FloatType),
      StructField("classIndex", FloatType)))
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelCol)

    val classifier = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol(labelCol)

    // Tune model using cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxDepth, Array(3, 8))
      .addGrid(classifier.eta, Array(0.2, 0.6))
      .build()

    val cv = new CrossValidator()
      .setEstimator(classifier)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val training = new GpuDataReader(ss).schema(schema).csv(getTestDataPath("/iris.csv"))
    val cvModel = cv.fit(training)
  }

  def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

}
