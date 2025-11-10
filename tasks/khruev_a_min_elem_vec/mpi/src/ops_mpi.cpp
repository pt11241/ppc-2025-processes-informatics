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

// bool KhruevAMinElemVecMPI::RunImpl() {
//   int rank, size;
//   // int local_size;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   int min = 0;
//   int global_min = 0;
//   int vector_size = GetInput().size();
//   std::vector<int> a(2);
//   if (size == 1) {
//     global_min = GetInput()[0];
//     for (int i = 0; i < vector_size; i++) {
//       if (GetInput()[i] < global_min) {
//         global_min = GetInput()[i];
//       }
//     }
//     GetOutput() = global_min;
//     return true;
//   }

//   if (rank == 0) {
//     SendingToProcceses(size, vector_size);  // отправка индексов процессам от 1 до n
//     // int global_min = 0;
//     bool have_global = false;
//     for (int i = 1; i <= size - 1; ++i) {
//       int local_min;
//       MPI_Recv(&local_min, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//       if (!have_global) {
//         global_min = local_min;
//         have_global = true;
//       } else {
//         global_min = std::min(global_min, local_min);
//       }
//     }
//   } else {
//     std::array<int, 2> range;
//     MPI_Recv(range.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     int start = range[0], end = range[1];
//     if (start > end) {
//       min = INT_MAX;
//       MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     } else {
//       min = GetInput()[start];
//       for (int i = start; i <= end; i++) {
//         if (GetInput()[i] < min) {
//           min = GetInput()[i];
//         }
//       }
//       MPI_Send(&min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }
//   }
//   MPI_Bcast(&global_min, 1, MPI_INT, 0, MPI_COMM_WORLD);

//   GetOutput() = global_min;
//   return true;
// }

bool KhruevAMinElemVecMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int vector_size = GetInput().size();
  int local_min = INT_MAX;
  int global_min = INT_MAX;

  // Handle single process case
  if (size == 1) {
    global_min = GetInput()[0];
    for (int i = 1; i < vector_size; i++) {
      if (GetInput()[i] < global_min) {
        global_min = GetInput()[i];
      }
    }
    GetOutput() = global_min;
    return true;
  }

  if (rank == 0) {
    // Send ranges to worker processes
    SendingToProcceses(size, vector_size);

    // Also compute local min for rank 0's portion
    // int workers = size - 1;
    int base = vector_size / size;
    int rem = vector_size % size;

    // Calculate rank 0's portion
    int chunk = base + (0 < rem ? 1 : 0);
    int start = 0;
    int end = (chunk > 0) ? (chunk - 1) : -1;

    if (start <= end && end < vector_size) {
      local_min = GetInput()[start];
      for (int i = start + 1; i <= end; i++) {
        if (GetInput()[i] < local_min) {
          local_min = GetInput()[i];
        }
      }
    }

    global_min = local_min;

    // Receive results from workers
    for (int i = 1; i < size; i++) {
      int worker_min;
      MPI_Recv(&worker_min, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (worker_min < global_min) {
        global_min = worker_min;
      }
    }

    // Send final result to all processes
    for (int i = 1; i < size; i++) {
      MPI_Send(&global_min, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
    }
  } else {
    // Worker processes
    std::array<int, 2> range;
    MPI_Recv(range.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start = range[0], end = range[1];
    if (start <= end && end < vector_size) {
      local_min = GetInput()[start];
      for (int i = start + 1; i <= end; i++) {
        if (GetInput()[i] < local_min) {
          local_min = GetInput()[i];
        }
      }
    }

    // Send local min to rank 0
    MPI_Send(&local_min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    // Receive final result from rank 0
    MPI_Recv(&global_min, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

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
