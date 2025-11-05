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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int global_size = GetInput().size();

  std::vector<int> sendcounts(size), displacements(size);
  int displacement = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = global_size / size + (i < global_size % size ? 1 : 0);
    displacements[i] = displacement;
    displacement += sendcounts[i];
  }

  int local_size = sendcounts[rank];
  std::vector<int> local_vec(local_size);

  MPI_Scatterv(GetInput().data(), sendcounts.data(), displacements.data(), MPI_INT, local_vec.data(), local_size,
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  if (!local_vec.empty()) {
    local_min = *std::min_element(local_vec.begin(), local_vec.end());
  }

  int global_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
