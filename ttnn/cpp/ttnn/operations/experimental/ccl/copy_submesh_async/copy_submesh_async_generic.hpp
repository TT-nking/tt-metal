// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteCopySubmeshAsyncGeneric {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshDevice* output_mesh_device,
        const MeshDevice* union_in_out_mesh_device,
        const uint32_t cluster_axis,
        std::optional<uint32_t> num_links,
        std::optional<ttnn::ccl::Topology> topology,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto copy_submesh_async_generic = ttnn::register_operation<
    "ttnn::experimental::copy_submesh_async_generic",
    ttnn::operations::experimental::ccl::ExecuteCopySubmeshAsyncGeneric>();

}  // namespace experimental
}  // namespace ttnn
