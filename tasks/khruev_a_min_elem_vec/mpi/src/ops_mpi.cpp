#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecMPI::KhruevAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask()); // mpi scoreboard
  GetInput() = in; // dannie doljna bit vidna vsem func rodytelya and stabilizaciya
  GetOutput() = 0;
}

bool KhruevAMinElemVecMPI::ValidationImpl() { // input check
  return !GetInput().empty();
}

bool KhruevAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecMPI::RunImpl() {
  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
