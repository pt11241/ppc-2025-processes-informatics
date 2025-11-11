#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecSEQ::KhruevAMinElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KhruevAMinElemVecSEQ::ValidationImpl() {
  return (!GetInput().empty());
}

bool KhruevAMinElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecSEQ::RunImpl() {
  int mininmum = GetInput()[0];
  size_t vec_size = GetInput().size();
  for (size_t i = 1; i < vec_size; i++) {
    if (GetInput()[i] < mininmum) {
      mininmum = GetInput()[i];
    }
  }
  GetOutput() = mininmum;

  return true;
}

bool KhruevAMinElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
