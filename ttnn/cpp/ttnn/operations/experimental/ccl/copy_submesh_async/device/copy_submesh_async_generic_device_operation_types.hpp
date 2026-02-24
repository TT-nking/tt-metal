// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <cstdint>
#include <tt-metalium/sub_device.hpp>
#include <optional>

namespace ttnn::experimental::prim {

struct CopySubmeshAsyncGenericParams {
    const distributed::MeshDevice* output_mesh_device;
    const distributed::MeshDevice* union_in_out_mesh_device;

    const uint32_t cluster_axis;
    const uint32_t num_links;
    const ttnn::ccl::Topology topology;
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
};

struct CopySubmeshAsyncGenericInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::experimental::prim
