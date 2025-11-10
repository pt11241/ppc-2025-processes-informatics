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
  // int local_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int min = 0;
  int global_min = 0;
  int vector_size = GetInput().size();
  // std::vector<int> a(2);
  if (size == 1) {
    global_min = GetInput()[0];
    for (int i = 0; i < vector_size; i++) {
      if (GetInput()[i] < global_min) {
        global_min = GetInput()[i];
      }
    }
    GetOutput() = global_min;
    return true;
  }

  if (rank == 0) {
    SendingToProcceses(size, vector_size);  // отправка индексов процессам от 1 до n
    // int global_min = 0;
    bool have_global = false;
    for (int i = 1; i <= size - 1; ++i) {
      int local_min;
      MPI_Recv(&local_min, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (!have_global) {
        global_min = local_min;
        have_global = true;
      } else {
        global_min = std::min(global_min, local_min);
      }
    }
  } else {
    std::array<int, 2> range;
    MPI_Recv(range.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int start = range[0], end = range[1];
    if (start > end) {
      min = INT_MAX;
      MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
      min = GetInput()[start];
      for (int i = start; i <= end; i++) {
        if (GetInput()[i] < min) {
          min = GetInput()[i];
        }
      }
      MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

void KhruevAMinElemVecMPI::SendingToProcceses(int size, int vector_size) {
  int workers = size - 1;
  if (workers <= 0) {
    return;  // нет рабочих
  }

  int base = vector_size / workers;
  int rem = vector_size % workers;
  int offset = 0;

  for (int i = 1; i <= workers; ++i) {
    int chunk = base + (i <= rem ? 1 : 0);
    int start = offset;
    int end = (chunk > 0) ? (offset + chunk - 1) : (offset - 1);  // если chunk==0 -> end < start
    std::array<int, 2> range{start, end};
    MPI_Send(range.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    offset += chunk;
  }
}

}  // namespace khruev_a_min_elem_vec
