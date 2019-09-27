// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>

#include "../helpers.h"
#include "../../../src/common/host_device_vector.h"

TEST(c_api, XGDMatrixCreateFromMatDT) {
  std::vector<int> col0 = {0, -1, 3};
  std::vector<float> col1 = {-4.0f, 2.0f, 0.0f};
  const char *col0_type = "int32";
  const char *col1_type = "float32";
  std::vector<void *> data = {col0.data(), col1.data()};
  std::vector<const char *> types = {col0_type, col1_type};
  DMatrixHandle handle;
  XGDMatrixCreateFromDT(data.data(), types.data(), 3, 2, &handle,
                        0);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, 2);
  ASSERT_EQ(info.num_row_, 3);
  ASSERT_EQ(info.num_nonzero_, 6);

  for (const auto &batch : (*dmat)->GetRowBatches()) {
    ASSERT_EQ(batch[0][0].fvalue, 0.0f);
    ASSERT_EQ(batch[0][1].fvalue, -4.0f);
    ASSERT_EQ(batch[2][0].fvalue, 3.0f);
    ASSERT_EQ(batch[2][1].fvalue, 0.0f);
  }

  delete dmat;
}

TEST(c_api, XGDMatrixCreateFromMat_omp) {
  std::vector<int> num_rows = {100, 11374, 15000};
  for (auto row : num_rows) {
    int num_cols = 50;
    int num_missing = 5;
    DMatrixHandle handle;
    std::vector<float> data(num_cols * row, 1.5);
    for (int i = 0; i < num_missing; i++) {
      data[i] = std::numeric_limits<float>::quiet_NaN();
    }

    XGDMatrixCreateFromMat_omp(data.data(), row, num_cols,
                               std::numeric_limits<float>::quiet_NaN(), &handle,
                               0);

    std::shared_ptr<xgboost::DMatrix> *dmat =
        static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
    xgboost::MetaInfo &info = (*dmat)->Info();
    ASSERT_EQ(info.num_col_, num_cols);
    ASSERT_EQ(info.num_row_, row);
    ASSERT_EQ(info.num_nonzero_, num_cols * row - num_missing);

    for (const auto &batch : (*dmat)->GetRowBatches()) {
      for (size_t i = 0; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (auto e : inst) {
          ASSERT_EQ(e.fvalue, 1.5);
        }
      }
    }
    delete dmat;
  }
}

#ifdef XGBOOST_USE_CUDF
// CUDF gdf_column wrapper that contains the pertinent data
struct GDFColumn {
  xgboost::HostDeviceVector<float> data;
  xgboost::HostDeviceVector<unsigned char> valid;
  gdf_column *gcol;

  GDFColumn() : gcol(new gdf_column) {}
  ~GDFColumn() { delete gcol; }

  GDFColumn &operator =(const GDFColumn &) = delete;
  GDFColumn &operator =(GDFColumn &&) = delete;
  GDFColumn(const GDFColumn &) = delete;
  GDFColumn(GDFColumn &&) = delete;
};

void
CreateGdfColumnMetaInfo(size_t begin_row, size_t n_rows, const xgboost::MetaInfo &minfo,
                        std::vector<GDFColumn> &gcols, int device_id) {
  CHECK_EQ(1, gcols.size());
  gcols[0].gcol->size = n_rows;
  gcols[0].gcol->dtype = GDF_FLOAT32;
  gcols[0].gcol->null_count = 0;

  // Create the data on host first and copy it to device next
  gcols[0].data.Reshard(xgboost::GPUDistribution(xgboost::GPUSet::All(device_id, 1)));
  gcols[0].data.Resize(n_rows);

  auto &data = gcols[0].data.HostVector();
  const auto &src_data = minfo.labels_.HostVector();
  data.insert(data.begin(), &src_data[begin_row], &src_data[begin_row + n_rows]);

  gcols[0].gcol->data = gcols[0].data.DevicePointer(device_id);
}

void
ConvertSparsePageToGdfColumns(const xgboost::SparsePage &sp, std::vector<GDFColumn> &gcols,
                              size_t batch_nrows, int device_id) {
  // Create a gdf_column
  for (size_t i = 0; i < gcols.size(); ++i) {
    gcols[i].gcol->size = batch_nrows;
    gcols[i].gcol->dtype = GDF_FLOAT32;

    auto inst = sp[i];
    gcols[i].gcol->null_count = batch_nrows - inst.size();

    // Create the data on host first and copy it to device next
    gcols[i].data.Reshard(xgboost::GPUDistribution(xgboost::GPUSet::All(device_id, 1)));
    gcols[i].data.Resize(batch_nrows);
    gcols[i].valid.Reshard(xgboost::GPUDistribution(xgboost::GPUSet::All(device_id, 1)));
    gcols[i].valid.Resize(((batch_nrows - 1) / 8) + 1);

    auto &data = gcols[i].data.HostVector();
    auto &valid = gcols[i].valid.HostVector();
    for (size_t j = 0; j < inst.size(); ++j) {
      size_t idx = inst[j].index;
      valid[idx / 8] |= (1 << (idx % 8));
      data[idx] = inst[j].fvalue;
    }

    gcols[i].gcol->data = gcols[i].data.DevicePointer(device_id);
    gcols[i].gcol->valid = gcols[i].valid.DevicePointer(device_id);
  }
}

TEST(c_api, XGDMatrixCreateFromCUDFTest) {
  // Create a DMatrix first for reference and build the same one through
  // CUDF API in batches and compare
  int constexpr kNRows = 1000, kNCols = 10;
  int constexpr device_id = 0;

  // Reference dmat
  std::unique_ptr<xgboost::DMatrix> ref_dmat(
    xgboost::CreateSparsePageDMatrixWithRC(kNRows, kNCols, 0, true));
  xgboost::SparsePage ref_dmat_page(*ref_dmat->GetRowBatches().begin());
  ref_dmat_page.SortRows();  // Sort rows as the test API may create features that are random

  // Reference dmat created through external memory API so that we can feed batches to the
  // CUDF API
  std::unique_ptr<xgboost::DMatrix> ext_dmat(
    xgboost::CreateSparsePageDMatrixWithRC(kNRows, kNCols, 128UL, true));

  const xgboost::MetaInfo &minfo = ext_dmat->Info();

  DMatrixHandle dmat_handle;
  bool first_batch = true;
  // Convert this dmatrix into columns by transposing each batch
  for (auto &batch : ext_dmat->GetRowBatches()) {
    batch.SortRows();  // Sort rows as the test API may create features that are random

    size_t batch_nrows = batch.Size();
    auto sp = batch.GetTranspose(minfo.num_col_);
    // Subsequent sparse pages will have the row indices offset by the number of
    // rows traversed thus far. Hence, adjust those indices by that offset to make
    // this batch look like a distinct/self contained sparse page
    std::for_each(sp.data.HostVector().begin(), sp.data.HostVector().end(),
                  [&](xgboost::Entry &ent) { ent.index -= batch.base_rowid; });

    std::vector<GDFColumn> gcols(minfo.num_col_);
    ConvertSparsePageToGdfColumns(sp, gcols, batch_nrows, device_id);
    std::vector<gdf_column *> cols;
    std::for_each(gcols.begin(), gcols.end(),
                  [&](const GDFColumn &col) { cols.push_back(col.gcol); });

    if (first_batch) {
      first_batch = false;
      ASSERT_EQ(0, XGDMatrixCreateFromCUDF(&cols[0], minfo.num_col_, &dmat_handle,
                                           device_id, std::nanf("")));
    } else {
      ASSERT_EQ(0, XGDMatrixAppendCUDF(&cols[0], minfo.num_col_, dmat_handle,
                                       device_id, std::nanf("")));
    }
  }

  // Check if the dmat tucked inside the handle matches the reference dmat
  const auto handle_dmat = *(static_cast<std::shared_ptr<xgboost::DMatrix> *>(dmat_handle));
  // Check metainfo first
  ASSERT_EQ(ref_dmat->Info().num_row_, handle_dmat->Info().num_row_);
  ASSERT_EQ(ref_dmat->Info().num_col_, handle_dmat->Info().num_col_);
  ASSERT_EQ(ref_dmat->Info().num_nonzero_, handle_dmat->Info().num_nonzero_);

  xgboost::SparsePage handle_page;
  size_t num_batches = 0;
  for (const auto &batch : handle_dmat->GetRowBatches()) {
    ++num_batches;
    handle_page = batch;
  }
  ASSERT_EQ(1, num_batches);

  // Compare offsets
  ASSERT_EQ(ref_dmat_page.offset.HostVector(), handle_page.offset.HostVector());

  // Compare the entries
  const auto &handle_entries = handle_page.data.HostVector();
  const auto &ref_entries = ref_dmat_page.data.HostVector();
  ASSERT_EQ(ref_entries.size(), handle_entries.size());
  for (size_t i = 0; i < ref_entries.size(); ++i) {
    ASSERT_EQ(ref_entries[i].index, handle_entries[i].index);
    ASSERT_EQ(ref_entries[i].fvalue, handle_entries[i].fvalue);
  }

  ASSERT_EQ(0, XGDMatrixFree(dmat_handle));
}

TEST(c_api, XGDMatrixXGDMatrixSetCUDFInfoTest) {
  int constexpr kNRows = 1000, kNCols = 10;
  int constexpr device_id = 0;

  // Reference dmat
  std::unique_ptr<xgboost::DMatrix> ref_dmat(
    xgboost::CreateSparsePageDMatrixWithRC(kNRows, kNCols, 0, true));
  xgboost::SparsePage ref_dmat_page(*ref_dmat->GetRowBatches().begin());
  ref_dmat_page.SortRows();  // Sort rows as the test API may create features that are random

  const xgboost::MetaInfo &ref_minfo = ref_dmat->Info();

  // Create a dmat handle first
  DMatrixHandle dmat_handle;
  {
    std::vector<GDFColumn> gcols(ref_minfo.num_col_);
    auto sp = ref_dmat_page.GetTranspose(ref_minfo.num_col_);
    ConvertSparsePageToGdfColumns(sp, gcols, ref_dmat_page.Size(), device_id);
    std::vector<gdf_column *> cols;
    std::for_each(gcols.begin(), gcols.end(),
                  [&](const GDFColumn &col) { cols.push_back(col.gcol); });
    ASSERT_EQ(0, XGDMatrixCreateFromCUDF(&cols[0], ref_minfo.num_col_, &dmat_handle,
                                         device_id, std::nanf("")));
  }

  // Now set the meta info through the XGDMatrixAppendCUDFInfo API and compare the
  // meta info from the handle to the one present in 'ref_dmat'
  for (size_t i = 0; i < ref_dmat_page.Size(); ++i) {
    std::vector<GDFColumn> gcols(1);
    std::vector<gdf_column *> cols;
    CreateGdfColumnMetaInfo(i, 1, ref_minfo, gcols, device_id);
    std::for_each(gcols.begin(), gcols.end(),
                  [&](const GDFColumn &col) { cols.push_back(col.gcol); });
    ASSERT_EQ(0, XGDMatrixAppendCUDFInfo(dmat_handle, "label", &cols[0], cols.size(), device_id));
  }

  // Check if the dmat meta info tucked inside the handle matches the reference dmat
  const auto handle_dmat = *(static_cast<std::shared_ptr<xgboost::DMatrix> *>(dmat_handle));
  ASSERT_EQ(ref_minfo.num_row_, handle_dmat->Info().num_row_);
  ASSERT_EQ(ref_minfo.num_col_, handle_dmat->Info().num_col_);
  ASSERT_EQ(ref_minfo.num_nonzero_, handle_dmat->Info().num_nonzero_);
  ASSERT_EQ(ref_minfo.labels_.HostVector(), handle_dmat->Info().labels_.HostVector());

  // Test the set API
  {
    std::vector<GDFColumn> gcols(1);
    std::vector<gdf_column *> cols;
    CreateGdfColumnMetaInfo(0, ref_dmat_page.Size(), ref_minfo, gcols, device_id);
    std::for_each(gcols.begin(), gcols.end(),
                  [&](const GDFColumn &col) { cols.push_back(col.gcol); });
    ASSERT_EQ(0, XGDMatrixSetCUDFInfo(dmat_handle, "label", &cols[0], cols.size(), device_id));
  }
  ASSERT_EQ(ref_minfo.num_row_, handle_dmat->Info().num_row_);
  ASSERT_EQ(ref_minfo.num_col_, handle_dmat->Info().num_col_);
  ASSERT_EQ(ref_minfo.num_nonzero_, handle_dmat->Info().num_nonzero_);
  ASSERT_EQ(ref_minfo.labels_.HostVector(), handle_dmat->Info().labels_.HostVector());

  ASSERT_EQ(0, XGDMatrixFree(dmat_handle));
}
#
#endif
