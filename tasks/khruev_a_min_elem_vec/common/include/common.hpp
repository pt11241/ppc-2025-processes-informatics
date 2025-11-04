#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace khruev_a_min_elem_vec {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace khruev_a_min_elem_vec
