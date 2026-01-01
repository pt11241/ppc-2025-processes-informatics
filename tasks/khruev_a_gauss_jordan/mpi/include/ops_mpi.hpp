#pragma once
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_gauss_jordan {

class KhruevAGaussJordanMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KhruevAGaussJordanMPI(const InType &in);

 private:
  std::vector<double> local_data_;
  int n_, m_;
  struct PivotPos {
    double val;
    int rank;
  };
  [[nodiscard]] int GetGlobalIdx(int local_k, int rank, int size) const;
  PivotPos FindPivot(int col, int rank, int size);
  void SwapRows(int i, int pivot_rank, int rank, int size);
  void Eliminate(int i, int rank, int size);
  static constexpr double kEps = 1e-10;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace khruev_a_gauss_jordan
