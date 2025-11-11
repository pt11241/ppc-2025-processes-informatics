#pragma once

#include <cstddef>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_min_elem_vec {

class KhruevAMinElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KhruevAMinElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void SendingToProcceses(int size, size_t vector_size);
  static void ReceivingAndReturn(std::vector<int> &vec);
};

}  // namespace khruev_a_min_elem_vec
