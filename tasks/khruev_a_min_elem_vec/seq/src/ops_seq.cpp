#include "khruev_a_min_elem_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <cstddef>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecSEQ::KhruevAMinElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KhruevAMinElemVecSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KhruevAMinElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecSEQ::RunImpl() {

  int mininmum = GetInput()[0];
  size_t vec_size = GetInput().size();
  for (size_t i = 1; i < vec_size; i++){
    if (GetInput()[i] < mininmum)
      mininmum = GetInput()[i];
  }
  GetOutput() = mininmum;

  return true;
}

bool KhruevAMinElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
