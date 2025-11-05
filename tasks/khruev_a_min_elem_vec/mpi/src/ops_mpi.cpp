#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecMPI::KhruevAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // mpi scoreboard
  GetInput() = in;                       // dannie doljna bit vidna vsem func rodytelya and stabilizaciya
  GetOutput() = 0;
}

bool KhruevAMinElemVecMPI::ValidationImpl() {  // input check
  return !GetInput().empty();
}

bool KhruevAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecMPI::RunImpl() {
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

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
