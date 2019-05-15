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

import ml.dmlc.xgboost4j.java.spark.nvidia.NVColumnBatch
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class NVDatasetSuite extends FunSuite with BeforeAndAfterAll {
  private val spark = SparkSession.builder.master("local").getOrCreate()

  override protected def afterAll(): Unit = spark.close()

  test("mapColumnarSingleBatchPerPartition") {
    val reader = new NVDataReader(spark)
    val dataPath = getTestDataPath("/rank.train.csv")
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(dataPath)
    val rdd = dataset.mapColumnarSingleBatchPerPartition((b: NVColumnBatch) =>
      Iterator.single(b.getNumColumns))
    val counts = rdd.collect
    assertResult(1) { counts.length }
    assertResult(5) { counts(0) }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
