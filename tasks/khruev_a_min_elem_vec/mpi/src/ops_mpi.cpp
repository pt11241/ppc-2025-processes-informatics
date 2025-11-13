#include "khruev_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "khruev_a_min_elem_vec/common/include/common.hpp"

namespace khruev_a_min_elem_vec {

KhruevAMinElemVecMPI::KhruevAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // mpi scoreboard
  GetInput() = in;                       // dannie doljna bit vidna vsem func rodytelya and stabilizaciya
  GetOutput() = 0;
}

bool KhruevAMinElemVecMPI::ValidationImpl() {  // input check
  return GetOutput() == 0;
}

bool KhruevAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool KhruevAMinElemVecMPI::RunImpl() {
  if (GetInput().empty()) {
    GetOutput() = INT_MAX;
    return true;
  }
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int global_min = 0;
  int local_min = 0;

  size_t vec_size = GetInput().size();

  size_t int_part = vec_size / size;
  size_t remainder = vec_size % size;

  size_t start = (int_part * rank) + std::min(static_cast<size_t>(rank), remainder);
  size_t end = start + int_part - 1 + (std::cmp_less(static_cast<size_t>(rank), remainder) ? 1 : 0);
  local_min = INT_MAX;
  if (start < vec_size) {
    local_min = GetInput()[start];
    for (size_t i = start; i <= end; i++) {
      local_min = std::min(local_min, GetInput()[i]);
    }
  }

  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
