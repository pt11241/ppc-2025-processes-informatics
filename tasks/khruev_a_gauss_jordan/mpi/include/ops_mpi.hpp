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
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void BroadcastSizes(int &rows, int &cols);
  static void BroadcastMatrix(std::vector<std::vector<double>> &mat, int rows, int cols);
};

}  // namespace khruev_a_gauss_jordan
