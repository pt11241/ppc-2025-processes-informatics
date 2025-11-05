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
  // int mininmum = GetInput()[0];
  // size_t vec_size = GetInput().size();
  // for (size_t i = 1; i < vec_size; i++) {
  //   if (GetInput()[i] < mininmum) {
  //     mininmum = GetInput()[i];
  //   }
  // }
  // GetOutput() = mininmum;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t vec_size = GetInput().size();
  int base_size = vec_size / size;
  int remainder = vec_size % size;
  int local_size = base_size + (rank < remainder ? 1 : 0);

  std::vector<int> local_vec(local_size);

  std::vector<int> sendcounts(size), displacements(size);
  if (rank == 0) {
    int displacement = 0;
    for (int i = 0; i < size; i++) {
      sendcounts[i] = base_size + (i < remainder ? 1 : 0);
      displacements[i] = displacement;
      displacement += sendcounts[i];
    }
  }

  MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(GetInput().data(), sendcounts.data(), displacements.data(), MPI_INT, local_vec.data(), local_size,
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  for (int value : local_vec) {
    if (value < local_min) {
      local_min = value;
    }
  }

  int global_min;
  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_min;
  }

  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_min_elem_vec
