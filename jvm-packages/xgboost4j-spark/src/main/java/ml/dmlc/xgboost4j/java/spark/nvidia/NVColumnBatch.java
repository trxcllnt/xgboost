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

package ml.dmlc.xgboost4j.java.spark.nvidia;

import ai.rapids.cudf.ColumnVector;

import ai.rapids.cudf.Table;
import org.apache.spark.sql.types.StructType;

public class NVColumnBatch {
  private final Table table;
  private final StructType schema;

  public NVColumnBatch(Table table, StructType schema) {
    this.table = table;
    this.schema = schema;
  }

  public StructType getSchema() {
    return schema;
  }

  public long getNumRows() {
    return table.getRowCount();
  }

  public int getNumColumns() {
    return table.getNumberOfColumns();
  }

  public long getColumn(int index) {
    ColumnVector v = table.getColumn(index);
    long nativeAddr = v.getNativeCudfColumnAddress();

    // Closing the vector to prevent a refcount leak.
    // Assumption is the cudf table tracked by this batch will always
    // outlive the use of the native address being returned here.
    v.close();

    return nativeAddr;
  }
}
