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

  int vector_size = GetInput().size();
  std::vector<int> a(2);
  int global_min = 0;
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
    SendingToProcceses(size, vector_size, a);  // отправка индексов процессам от 1 до n

    for (int i = 1; i <= size - 1; i++) {
      MPI_Recv(&min, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (i == 1) {
        global_min = min;
      }
      if (global_min > min) {
        global_min = min;
      }
    }

  } else {
    MPI_Recv(a.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int start = a[0], end = a[1];
    min = GetInput()[start];
    for (int i = start; i <= end; i++) {
      if (GetInput()[i] < min) {
        min = GetInput()[i];
      }
    }

    MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool KhruevAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

void KhruevAMinElemVecMPI::SendingToProcceses(int size, int vector_size, std::vector<int> &a) {
  int elem_num = vector_size / (size - 1);
  int remainder = vector_size % (size - 1);
  int flag = elem_num;
  for (int i = 1; i <= size - 1; i++) {
    if (flag == 0) {
      elem_num++;
      flag = 1;
    }
    if (i == (size - 1) && remainder != 0) {
      a[0] = (i - 1) * elem_num;
      a[1] = vector_size - 1;
    } else {
      a[0] = (i - 1) * elem_num;
      a[1] = i * elem_num - 1;
    }
    MPI_Send(a.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
  }
}

}  // namespace khruev_a_min_elem_vec
