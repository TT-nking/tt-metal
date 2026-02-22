// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace nb = nanobind;
void bind_copy_submesh_async_generic(nb::module_& mod);

}  // namespace ttnn::operations::experimental::ccl
