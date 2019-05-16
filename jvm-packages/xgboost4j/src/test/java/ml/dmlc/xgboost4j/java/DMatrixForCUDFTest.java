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
package ml.dmlc.xgboost4j.java;

import ai.rapids.cudf.ColumnVector;
import junit.framework.TestCase;
import org.junit.Test;

import java.util.Arrays;

/**
 * test cases for DMatrix using CUDF
 *
 * @author liangcail
 */
public class DMatrixForCUDFTest {

  @Test
  public void testCreateFromCUDF() {
    //create Matrix from CUDF
    ColumnVector featureCol = null, labelCol = null, weightCol = null;
    try {
      float[] infoData = new float[]{5.0f, 6.0f, 7.0f};
      // feature
      featureCol = ColumnVector.fromFloats(1.0f, 2.0f, 3.0f);
      featureCol.ensureOnDevice();
      // label
      labelCol = ColumnVector.fromFloats(infoData);
      labelCol.ensureOnDevice();
      // weight
      weightCol = ColumnVector.fromFloats(infoData);
      weightCol.ensureOnDevice();

      DMatrix dmat = new DMatrix(new long[]{featureCol.getNativeCudfColumnAddress()});
      dmat.setCUDFInfo("label", new long[]{labelCol.getNativeCudfColumnAddress()});
      dmat.setCUDFInfo("weight", new long[]{weightCol.getNativeCudfColumnAddress()});

      TestCase.assertTrue("Wrong label ", Arrays.equals(dmat.getLabel(), infoData));
      TestCase.assertTrue("Wrong weight ", Arrays.equals(dmat.getWeight(), infoData));
    } catch (XGBoostError xe) {
      // In case CUDF is not built
      TestCase.assertTrue("Unexpected error: " + xe,
            "CUDF is not enabled!".equals(xe.getMessage()));
    } finally {
      if (featureCol != null) featureCol.close();
      if (labelCol != null) labelCol.close();
      if (weightCol != null) weightCol.close();
    }
  }
}
