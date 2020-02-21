#
# Copyright (c) 2019 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import pyspark.ml.tuning

from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.util import _jvm
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

class CrossValidator(JavaWrapper, pyspark.ml.tuning.CrossValidator):

    def __init__(self):
        super(CrossValidator, self).__init__()
        self._java_obj = self._new_java_obj(
            'ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator')

    def fit(self, dataset):
        java_estimator, java_epms, java_evaluator = self._to_java_impl()
        self._java_obj.setEstimator(java_estimator)
        self._java_obj.setEvaluator(java_evaluator)
        self._java_obj.setEstimatorParamMaps(java_epms)

        dataset = dataset._jdf if isinstance(dataset, DataFrame) else dataset._java_obj
        java_model = self._java_obj.fit(dataset)
        return self.getEstimator()._create_model(java_model)

class GpuDataset(JavaWrapper):
    # Note: SparkSession.getActiveSession() could be used in Spark 3.x
    def __init__(self, spark_session, java_gpu_dataset):
        java_gpu_dataset._jdf = java_gpu_dataset
        java_gpu_dataset.sql_ctx = spark_session
        super(GpuDataset, self).__init__(java_gpu_dataset)

    def schema(self):
        return StructType.fromJson(json.loads(self._java_obj.schema().json()))

class GpuDataReader(JavaWrapper):
    # Note: SparkSession.getActiveSession() could be used in Spark 3.x
    def __init__(self, spark_session):
        super(GpuDataReader, self).__init__()
        self._java_obj = self._new_java_obj(
            'ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader',
            spark_session._jsparkSession)
        self._spark_session = spark_session

    def format(self, source):
        self._java_obj.format(source)
        return self

    # schema is either a StructType or a String
    def schema(self, schema):
        if isinstance(schema, StructType):
            schema = _jvm().org.apache.spark.sql.types.StructType.fromJson(schema.json())
        self._java_obj.schema(schema)
        return self

    def option(self, key, value):
        self._java_obj.option(key, value)
        return self

    # options is of type Dict[String, String]
    def options(self, options):
        self._java_obj.options(options)
        return self

    def load(self, *paths):
        if len(paths) == 0:
            java_dataset = self._java_obj.load()
        elif len(paths) == 1:
            java_dataset = self._java_obj.load(paths[0])
        else:
            java_dataset = self._java_obj.load(_jvm().PythonUtils.toSeq(paths))
        return GpuDataset(self._spark_session, java_dataset)

    def csv(self, *paths):
        paths = paths[0] if len(paths) == 1 else _jvm().PythonUtils.toSeq(paths)
        return GpuDataset(self._spark_session, self._java_obj.csv(paths))

    def parquet(self, *paths):
        paths = paths[0] if len(paths) == 1 else _jvm().PythonUtils.toSeq(paths)
        return GpuDataset(self._spark_session, self._java_obj.parquet(paths))

    def orc(self, *paths):
        paths = paths[0] if len(paths) == 1 else _jvm().PythonUtils.toSeq(paths)
        return GpuDataset(self._spark_session, self._java_obj.orc(paths))
