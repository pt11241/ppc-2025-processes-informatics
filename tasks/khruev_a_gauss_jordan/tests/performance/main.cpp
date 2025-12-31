#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "khruev_a_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace khruev_a_gauss_jordan {

class KhruevGaussJordanPerfBase : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input;

  void GenerateMatrix(int size, double diag_value) {
    input.assign(static_cast<size_t>(size), std::vector<double>(static_cast<size_t>(size + 1), 0.0));

    for (int i = 0; i < size; ++i) {
      double rhs = 0.0;
      for (int j = 0; j < size; ++j) {
        if (i == j) {
          input[i][j] = diag_value + (i % 7);
        } else {
          input[i][j] = (i + j) % 5 * 0.1;
        }
        rhs += std::abs(input[i][j]);
      }
      input[i][size] = rhs;
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    return !output.empty();
  }

  InType GetTestInputData() final {
    return input;
  }
};

class KhruevGaussJordanPerfSmall : public KhruevGaussJordanPerfBase {
  void SetUp() override {
    GenerateMatrix(12, 4.0);
  }
};

class KhruevGaussJordanPerfMedium : public KhruevGaussJordanPerfBase {
  void SetUp() override {
    GenerateMatrix(40, 8.0);
  }
};

class KhruevGaussJordanPerfDiagonal : public KhruevGaussJordanPerfBase {
  void SetUp() override {
    int n = 60;
    input.assign(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n + 1), 0.0));

    for (int i = 0; i < n; ++i) {
      input[i][i] = 10.0 + (i % 5);
      input[i][n] = input[i][i] * (i + 1);
    }
  }
};

TEST_P(KhruevGaussJordanPerfSmall, RunPerf) {
  ExecuteTest(GetParam());
}

TEST_P(KhruevGaussJordanPerfMedium, RunPerf) {
  ExecuteTest(GetParam());
}

TEST_P(KhruevGaussJordanPerfDiagonal, RunPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasksSmall = ppc::util::MakeAllPerfTasks<InType, KhruevAGaussJordanMPI, KhruevAGaussJordanSEQ>(
    PPC_SETTINGS_khruev_a_gauss_jordan);

const auto kPerfTasksMedium = ppc::util::MakeAllPerfTasks<InType, KhruevAGaussJordanMPI, KhruevAGaussJordanSEQ>(
    PPC_SETTINGS_khruev_a_gauss_jordan);

const auto kPerfTasksDiagonal = ppc::util::MakeAllPerfTasks<InType, KhruevAGaussJordanMPI, KhruevAGaussJordanSEQ>(
    PPC_SETTINGS_khruev_a_gauss_jordan);

const auto kValuesSmall = ppc::util::TupleToGTestValues(kPerfTasksSmall);

const auto kValuesMedium = ppc::util::TupleToGTestValues(kPerfTasksMedium);

const auto kValuesDiagonal = ppc::util::TupleToGTestValues(kPerfTasksDiagonal);

const auto kNameSmall = KhruevGaussJordanPerfSmall::CustomPerfTestName;

const auto kNameMedium = KhruevGaussJordanPerfMedium::CustomPerfTestName;

const auto kNameDiagonal = KhruevGaussJordanPerfDiagonal::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KhruevPerfSmall, KhruevGaussJordanPerfSmall, kValuesSmall, kNameSmall);

INSTANTIATE_TEST_SUITE_P(KhruevPerfMedium, KhruevGaussJordanPerfMedium, kValuesMedium, kNameMedium);

INSTANTIATE_TEST_SUITE_P(KhruevPerfDiagonal, KhruevGaussJordanPerfDiagonal, kValuesDiagonal, kNameDiagonal);

}  // namespace khruev_a_gauss_jordan
