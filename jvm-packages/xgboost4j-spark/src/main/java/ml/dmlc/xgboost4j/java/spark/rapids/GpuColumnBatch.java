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

package ml.dmlc.xgboost4j.java.spark.rapids;

import java.util.ArrayList;
import java.util.List;

import ai.rapids.cudf.ColumnVector;

import ai.rapids.cudf.Table;
import org.apache.spark.sql.types.StructType;

public class GpuColumnBatch {
  private final Table table;
  private final StructType schema;

  public GpuColumnBatch(Table table, StructType schema) {
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

  public ColumnVector getColumnVector(int index) {
    return table.getColumn(index);
  }

  public long getColumn(int index) {
    ColumnVector v = table.getColumn(index);
    return v.getNativeCudfColumnAddress();
  }

  public int[] groupByColumnWithCountHost(int groupIndex) {
    ColumnVector cv = getColumnVector(groupIndex);
    cv.ensureOnHost();
    List<Integer> countData = new ArrayList<>();
    int groupId = 0, groupSize = 0;
    for (int i = 0; i < cv.getRowCount(); i ++) {
      if(groupId != cv.getInt(i)) {
        groupId = cv.getInt(i);
        if (groupSize > 0) {
          countData.add(groupSize);
        }
        groupSize = 1;
      } else {
        groupSize ++;
      }
    }
    if (groupSize > 0) {
      countData.add(groupSize);
    }
    int[] counts = new int[countData.size()];
    for (int i = 0; i < counts.length; i ++) {
      counts[i] = countData.get(i);
    }
    return counts;
  }

  /**
   * This is used to group the CUDF Dataset by column 'groupIndex', and merge all the rows into
   * one row in each group based on column 'oneIndex'. For example
   * 1 0
   * 2 3   (0, 1, true)   0
   * 2 3        =>        3
   * 5 6                  6
   * 5 6
   * And the last row in each group is selected.
   * @param groupIndex The index of column to group by
   * @param oneIndex The index of column to get one value in each group
   * @param checkEqual Whether to check all the values in one group are the same
   * @return The native handle of the result column containing the merged data.
   */
  public long[] groupByColumnWithAggregation(int groupIndex, int oneIndex, boolean checkEqual) {
    ColumnVector cv = getColumnVector(groupIndex);
    cv.ensureOnHost();
    ColumnVector aggrCV = getColumnVector(oneIndex);
    aggrCV.ensureOnHost();
    List<Float> onesData = new ArrayList<>();
    int groupId = 0;
    Float oneValue = null;
    for (int i = 0; i < cv.getRowCount(); i ++) {
      if(groupId != cv.getInt(i)) {
        groupId = cv.getInt(i);
        if (oneValue != null) {
          onesData.add(oneValue);
        }
        oneValue = aggrCV.getFloat(i);
      } else {
        if (checkEqual && oneValue != null && oneValue != aggrCV.getFloat(i)) {
          return null;
        }
      }
    }
    if (oneValue != null) {
      onesData.add(oneValue);
    }
    // FIXME how to release this ColumnVector?
    ColumnVector retCV = ColumnVector.fromBoxedFloats(onesData.toArray(new Float[]{}));
    retCV.ensureOnDevice();
    return new long[] {retCV.getNativeCudfColumnAddress()};
  }
}
