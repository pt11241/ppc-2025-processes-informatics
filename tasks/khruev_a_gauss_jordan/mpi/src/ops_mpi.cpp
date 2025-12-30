#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace khruev_a_gauss_jordan {

namespace {

constexpr double kEps = 1e-10;

bool IsNearlyZero(double v) {
  return std::fabs(v) < kEps;
}

int ChoosePivot(const std::vector<std::vector<double>> &a, int col, int from, int n) {
  int best = -1;
  double max_val = 0.0;

  for (int i = from; i < n; ++i) {
    double cur = std::fabs(a[i][col]);
    if (cur > max_val) {
      max_val = cur;
      best = i;
    }
  }
  return (max_val > kEps) ? best : -1;
}

void NormalizeRow(std::vector<double> &row, int pivot_col) {
  double div = row[pivot_col];
  if (IsNearlyZero(div)) {
    return;
  }
  for (double &x : row) {
    x /= div;
  }
}

void SubtractRows(std::vector<double> &target, const std::vector<double> &pivot, double factor) {
  for (size_t j = 0; j < target.size(); ++j) {
    target[j] -= factor * pivot[j];
  }
}

void SolveGaussJordanSequential(std::vector<std::vector<double>> &a) {
  int n = static_cast<int>(a.size());
  int m = static_cast<int>(a[0].size());
  int row = 0;

  for (int col = 0; col < m - 1 && row < n; ++col) {
    int pivot = -1;
    pivot = ChoosePivot(a, col, row, n);

    if (pivot == -1) {
      continue;
    }

    std::swap(a[row], a[pivot]);
    NormalizeRow(a[row], col);

    for (int i = 0; i < n; ++i) {
      if (i == row) {
        continue;
      }

      double coeff = a[i][col];
      if (!IsNearlyZero(coeff)) {
        SubtractRows(a[i], a[row], coeff);
      }
    }

    ++row;
  }
}

std::vector<double> ExtractSolution(const std::vector<std::vector<double>> &a) {
  int n = static_cast<int>(a.size());
  int m = static_cast<int>(a[0].size());

  std::vector<double> solution(m - 1, 0.0);

  for (int i = 0; i < n; ++i) {
    int lead = -1;

    for (int j = 0; j < m - 1; ++j) {
      if (!IsNearlyZero(a[i][j])) {
        lead = j;
        break;
      }
    }

    if (lead != -1) {
      solution[lead] = a[i][m - 1];
    }
  }

  return solution;
}

}  // namespace

KhruevAGaussJordanMPI::KhruevAGaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  khruev_a_gauss_jordan::InType tmp(in);
  GetInput().swap(tmp);
}

bool KhruevAGaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ok = 1;
  if (rank == 0) {
    if (GetInput().empty()) {
      ok = 1;
    } else {
      size_t cols = GetInput()[0].size();
      for (const auto &r : GetInput()) {
        if (r.size() != cols) {
          ok = 0;
        }
      }
    }
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok == 1;
}

void KhruevAGaussJordanMPI::BroadcastSizes(int &rows, int &cols) {
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void KhruevAGaussJordanMPI::BroadcastMatrix(std::vector<std::vector<double>> &mat, int rows, int cols) {
  std::vector<double> buf(rows * cols);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        buf[(i * cols) + j] = mat[i][j];
      }
    }
  }

  MPI_Bcast(buf.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    mat.assign(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        mat[i][j] = buf[(i * cols) + j];
      }
    }
  }
}

bool KhruevAGaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetOutput().clear();

  int n = 0;
  int m = 0;
  if (rank == 0 && !GetInput().empty()) {
    n = static_cast<int>(GetInput().size());
    m = static_cast<int>(GetInput()[0].size());
  }

  BroadcastSizes(n, m);
  if (n == 0 || m == 0) {
    return true;
  }

  BroadcastMatrix(GetInput(), n, m);
  return true;
}

bool KhruevAGaussJordanMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (GetInput().empty()) {
    return true;
  }

  int m = static_cast<int>(GetInput()[0].size());
  std::vector<double> solution(m - 1, 0.0);

  if (rank == 0) {
    auto a = GetInput();
    SolveGaussJordanSequential(a);
    solution = ExtractSolution(a);
  }

  MPI_Bcast(solution.data(), m - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = solution;
  return true;
}

bool KhruevAGaussJordanMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_gauss_jordan
