#include "khruev_a_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "khruev_a_gauss_jordan/common/include/common.hpp"

namespace khruev_a_gauss_jordan {

int KhruevAGaussJordanMPI::GetGlobalIdx(int local_k, int rank, int size) const {
  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  if (rank < remainder) {
    return (rank * (rows_per_proc + 1)) + local_k;
  }
  return (remainder * (rows_per_proc + 1)) + ((rank - remainder) * rows_per_proc) + local_k;
}

KhruevAGaussJordanMPI::RowPos KhruevAGaussJordanMPI::GetRowOwner(int global_i, int size) const {
  int rows = n_ / size;
  int rem = n_ % size;

  if (global_i < rem * (rows + 1)) {
    return {global_i / (rows + 1), global_i % (rows + 1)};
  }

  int base = rem * (rows + 1);
  return {rem + (global_i - base) / rows, (global_i - base) % rows};
}

int KhruevAGaussJordanMPI::FindLocalPivotIdx(int col, int rank, int size) const {
  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? rows + 1 : rows;

  int best = -1;
  double best_val = -1.0;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) >= col) {
      double v = std::fabs(local_data_[k * m_ + col]);
      if (v > best_val) {
        best_val = v;
        best = k;
      }
    }
  }
  return best;
}

KhruevAGaussJordanMPI::PivotPos KhruevAGaussJordanMPI::FindPivot(int col, int rank, int size) {
  PivotPos local_piv = {.val = -1.0, .rank = rank};
  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  int my_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) >= col) {
      double val = std::fabs(local_data_[(k * m_) + col]);
      local_piv.val = std::max(val, local_piv.val);
    }
  }

  PivotPos global_piv{};
  MPI_Allreduce(&local_piv, &global_piv, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  return global_piv;
}

void KhruevAGaussJordanMPI::SwapLocalRows(int a, int b) {
  if (a == b) {
    return;
  }
  std::swap_ranges(local_data_.data() + a * m_, local_data_.data() + (a + 1) * m_, local_data_.data() + b * m_);
}

void KhruevAGaussJordanMPI::SwapRemoteRows(int my_idx, int other_rank) {
  std::vector<double> tmp(m_);

  MPI_Sendrecv(local_data_.data() + my_idx * m_, m_, MPI_DOUBLE, other_rank, 0, tmp.data(), m_, MPI_DOUBLE, other_rank,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::ranges::copy(tmp, local_data_.data() + my_idx * m_);
}

void KhruevAGaussJordanMPI::NormalizePivotRow(int i, const RowPos &pivot, std::vector<double> &pivot_row, int rank) {
  if (rank != pivot.rank) {
    return;
  }

  double *row = local_data_.data() + pivot.local_idx * m_;
  double div = row[i];

  if (std::fabs(div) > kEps) {
    for (int j = i; j < m_; ++j) {
      row[j] /= div;
    }
  }
  std::ranges::copy(row, row + m_, pivot_row.begin());
}

void KhruevAGaussJordanMPI::ApplyElimination(int i, const std::vector<double> &pivot_row, int rank, int size) {
  int rows = n_ / size;
  int rem = n_ % size;
  int my_rows = (rank < rem) ? rows + 1 : rows;

  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) == i) {
      continue;
    }

    double *row = local_data_.data() + k * m_;
    double factor = row[i];

    if (std::fabs(factor) > kEps) {
      for (int j = i; j < m_; ++j) {
        row[j] -= factor * pivot_row[j];
      }
    }
  }
}

void KhruevAGaussJordanMPI::SwapRows(int i, int pivot_rank, int rank, int size) {
  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  int target_rank = (i < remainder * (rows_per_proc + 1))
                        ? i / (rows_per_proc + 1)
                        : remainder + ((i - remainder * (rows_per_proc + 1)) / rows_per_proc);
  int target_local_i = (i < remainder * (rows_per_proc + 1)) ? i % (rows_per_proc + 1)
                                                             : (i - remainder * (rows_per_proc + 1)) % rows_per_proc;

  // Ищем локальный индекс строки с макс. значением на процессе-владельце
  int local_pivot_idx = -1;
  if (rank == pivot_rank) {
    int my_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;
    double max_val = -1.0;
    for (int k = 0; k < my_rows; ++k) {
      if (GetGlobalIdx(k, rank, size) >= i) {
        double val = std::fabs(local_data_[(k * m_) + i]);
        if (val > max_val) {
          max_val = val;
          local_pivot_idx = k;
        }
      }
    }
  }

  if (pivot_rank == target_rank) {
    if (rank == target_rank && local_pivot_idx != target_local_i) {
      std::swap_ranges(local_data_.begin() + static_cast<size_t>(local_pivot_idx) * m_,
                       local_data_.begin() + static_cast<size_t>((static_cast<size_t>(local_pivot_idx) + 1) * m_),
                       local_data_.begin() + static_cast<size_t>(static_cast<size_t>(target_local_i) * m_));
    }
  } else {
    std::vector<double> tmp(m_);
    if (rank == pivot_rank) {
      MPI_Sendrecv(&local_data_[static_cast<size_t>(local_pivot_idx) * m_], m_, MPI_DOUBLE, target_rank, 0, tmp.data(),
                   m_, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::copy(tmp.begin(), tmp.end(), &local_data_[static_cast<size_t>(local_pivot_idx) * m_]);
    } else if (rank == target_rank) {
      MPI_Sendrecv(&local_data_[static_cast<size_t>(target_local_i) * m_], m_, MPI_DOUBLE, pivot_rank, 0, tmp.data(),
                   m_, MPI_DOUBLE, pivot_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::copy(tmp.begin(), tmp.end(), &local_data_[static_cast<size_t>(target_local_i) * m_]);
    }
  }
}

void KhruevAGaussJordanMPI::Eliminate(int i, int rank, int size) {
  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  int target_rank = (i < remainder * (rows_per_proc + 1))
                        ? i / (rows_per_proc + 1)
                        : remainder + ((i - remainder * (rows_per_proc + 1)) / rows_per_proc);
  int target_local_i = (i < remainder * (rows_per_proc + 1)) ? i % (rows_per_proc + 1)
                                                             : (i - remainder * (rows_per_proc + 1)) % rows_per_proc;

  std::vector<double> pivot_row(m_);
  if (rank == target_rank) {
    double divisor = local_data_[(target_local_i * m_) + i];
    if (std::fabs(divisor) > kEps) {
      for (int j = i; j < m_; ++j) {
        local_data_[(target_local_i * m_) + j] /= divisor;
      }
    }
    std::copy(local_data_.begin() + static_cast<size_t>(target_local_i) * m_,
              local_data_.begin() + static_cast<size_t>(target_local_i + 1) * m_, pivot_row.begin());
  }

  MPI_Bcast(pivot_row.data(), m_, MPI_DOUBLE, target_rank, MPI_COMM_WORLD);

  int my_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  for (int k = 0; k < my_rows; ++k) {
    if (GetGlobalIdx(k, rank, size) != i) {
      double factor = local_data_[(k * m_) + i];
      if (std::fabs(factor) > kEps) {
        for (int j = i; j < m_; ++j) {
          local_data_[(k * m_) + j] -= factor * pivot_row[j];
        }
      }
    }
  }
}

KhruevAGaussJordanMPI::KhruevAGaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // Важно: на вход в конструктор данные могут прийти только на Rank 0 в тестах
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

bool KhruevAGaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ok = 1;
  if (rank == 0) {
    if (GetInput().empty() || GetInput()[0].empty()) {
      ok = 0;
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok == 1;
}

bool KhruevAGaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    n_ = static_cast<int>(GetInput().size());
    m_ = static_cast<int>(GetInput()[0].size());
  }
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  int my_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;

  local_data_.assign(static_cast<size_t>(my_rows) * m_, 0.0);

  // Подготовка Scatterv
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = ((i < remainder) ? (rows_per_proc + 1) : rows_per_proc) * m_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<double> full_matrix_flat;
  if (rank == 0) {
    full_matrix_flat.resize(static_cast<size_t>(n_) * m_);
    for (int i = 0; i < n_; ++i) {
      std::copy(GetInput()[i].begin(), GetInput()[i].end(), full_matrix_flat.begin() + static_cast<size_t>(i) * m_);
    }
  }

  MPI_Scatterv(full_matrix_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data_.data(), my_rows * m_,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool KhruevAGaussJordanMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int i = 0; i < n_; ++i) {
    PivotPos global_piv = FindPivot(i, rank, size);
    if (global_piv.val < kEps) {
      continue;
    }

    SwapRows(i, global_piv.rank, rank, size);
    Eliminate(i, rank, size);
  }
  return true;
}

bool KhruevAGaussJordanMPI::PostProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_proc = n_ / size;
  int remainder = n_ % size;
  int my_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;

  std::vector<double> full_results;
  if (rank == 0) {
    full_results.resize(static_cast<size_t>(n_) * m_);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = ((i < remainder) ? (rows_per_proc + 1) : rows_per_proc) * m_;
    displs[i] = offset;
    offset += recvcounts[i];
  }

  MPI_Gatherv(local_data_.data(), my_rows * m_, MPI_DOUBLE, full_results.data(), recvcounts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  OutType final_solution(n_);
  if (rank == 0) {
    for (int i = 0; i < n_; ++i) {
      final_solution[i] = full_results[(i * m_) + (m_ - 1)];
    }
  }

  // РЕШАЮЩИЙ МОМЕНТ: Рассылаем ответ всем, чтобы CheckTestOutputData прошел успешно
  MPI_Bcast(final_solution.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_solution;
  return true;
}

}  // namespace khruev_a_gauss_jordan
