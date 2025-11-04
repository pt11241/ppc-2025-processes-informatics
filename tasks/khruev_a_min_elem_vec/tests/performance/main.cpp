#include <gtest/gtest.h>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KhruevAMinElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KhruevAMinElemVecMPI, KhruevAMinElemVecSEQ>(PPC_SETTINGS_khruev_a_min_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KhruevAMinElemVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KhruevAMinElemVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace khruev_a_min_elem_vec
