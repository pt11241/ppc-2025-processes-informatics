#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    size_t size = 10;
    for (size_t i = 1; i <= size; i++) {
      input_data_.push_back(i);
    }
    expected_ = 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(KhruevAMinElemVecFuncTests, MinElemVecc) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {"aaaaaaa"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KhruevAMinElemVecMPI, InType>(kTestParam, PPC_SETTINGS_khruev_a_min_elem_vec),
    ppc::util::AddFuncTask<KhruevAMinElemVecSEQ, InType>(kTestParam, PPC_SETTINGS_khruev_a_min_elem_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KhruevAMinElemVecFuncTests::PrintFuncTestName<KhruevAMinElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinElemVec, KhruevAMinElemVecFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace khruev_a_min_elem_vec
