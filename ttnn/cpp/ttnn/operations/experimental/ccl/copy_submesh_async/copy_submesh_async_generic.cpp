// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_submesh_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/copy_submesh_async/device/copy_submesh_async_generic_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteCopySubmeshAsyncGeneric::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshDevice* output_mesh_device,
    const MeshDevice* union_in_out_mesh_device,
    const uint32_t cluster_axis,
    std::optional<uint32_t> num_links,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    auto* input_mesh_device = input_tensor.device();
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    uint32_t num_links_ = num_links.value_or(ttnn::operations::ccl::common::get_num_links(*input_mesh_device));

    return ttnn::prim::copy_submesh_async_generic(
        input_tensor, output_mesh_device, union_in_out_mesh_device, cluster_axis, num_links_, topology_, subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
