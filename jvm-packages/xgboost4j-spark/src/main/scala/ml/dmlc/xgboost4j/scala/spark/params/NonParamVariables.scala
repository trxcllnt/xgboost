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

package ml.dmlc.xgboost4j.scala.spark.params

import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataset
import org.apache.spark.sql.DataFrame

trait NonParamVariables {
  protected var evalSetsMap: Map[String, DataFrame] = Map.empty

  def setEvalSets(evalSets: Map[String, DataFrame]): this.type = {
    evalSetsMap = evalSets
    this
  }

  def getEvalSets(params: Map[String, Any]): Map[String, DataFrame] = {
    if (params.contains("eval_sets")) {
      params("eval_sets").asInstanceOf[Map[String, DataFrame]]
    } else {
      evalSetsMap
    }
  }

  protected var gpuEvalSetsMap: Map[String, GpuDataset] = Map.empty

  def setGpuEvalSets(evalSets: Map[String, GpuDataset]): this.type = {
    gpuEvalSetsMap = evalSets
    this
  }

  def getGpuEvalSets(params: Map[String, Any]): Map[String, GpuDataset] = {
    // To minimize the change in apps, we share the same "eval_sets" with cpu.
    // Then NO code update is needed for the eval parameter part, just get the eval
    // sets as GpuDatasets.
    if (params.contains("eval_sets")) {
      val evals = params("eval_sets").asInstanceOf[Map[String, GpuDataset]]
      // Do value type check here because the above just checks the first layer: Map,
      // even specifying the types for both key and value.
      require(evals.values.forall(_.isInstanceOf[GpuDataset]),
        "Wrong type for value! Evaluation sets should be Map(name: String -> GpuDataset) for GPU.")
      evals.asInstanceOf[Map[String, GpuDataset]]
    } else {
      gpuEvalSetsMap
    }
  }
}
