// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_submesh_async_generic_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/copy_submesh_async/copy_submesh_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_copy_submesh_async_generic_op(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MeshDevice* output_mesh_device,
               const MeshDevice* union_in_out_mesh_device,
               const uint32_t cluster_axis,
               std::optional<uint32_t> num_links,
               std::optional<ttnn::ccl::Topology> topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    output_mesh_device,
                    union_in_out_mesh_device,
                    cluster_axis,
                    num_links,
                    topology,
                    subdevice_id);
            },
            nb::arg("input_tensor"),
            nb::arg("output_mesh_device"),
            nb::arg("union_in_out_mesh_device"),
            nb::arg("cluster_axis"),
            nb::kw_only(),
            nb::arg("num_links") = nb::none(),
            nb::arg("topology") = nb::none(),
            nb::arg("subdevice_id") = nb::none()});
}

}  // namespace

void bind_copy_submesh_async_generic(nb::module_& mod) {
    bind_copy_submesh_async_generic_op(
        mod,
        ttnn::experimental::copy_submesh_async_generic,
        R"doc(
        Performs an asynchronous all-to-all collective communication operation across multiple devices.

        This operation redistributes data between devices by having each device split its input tensor
        into N chunks (where N is the number of devices) and sending the i-th chunk to device i.
        The operation uses asynchronous kernels for improved performance.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to redistribute.
            in_dim (int): The dimension number to split.
            out_dim (int): The dimension number to concatenate.

        Keyword Args:
            persistent_output_buffer (ttnn.Tensor, optional): Buffer where final output will be written.
            num_links (int, optional): Number of fabric links to use for communication.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for buffers.
            topology (ttnn.Topology, optional): Network topology to use. Defaults to ttnn.Topology.Ring.
            subdevice_id (SubDeviceId, optional): Target specific subdevice for the operation.
            cluster_axis (int, optional): The axis along which to cluster the communication.

        Returns:
            ttnn.Tensor: Output tensor containing redistributed data.

        Example:
            >>> # Redistribute data from dim 2 to dim 3 across 4 devices
            >>> output_buf = ttnn.zeros_like(input_tensor)       # Output buffer
            >>> result = ttnn.experimental.copy_submesh_async_generic(
            ...     input_tensor,
            ...     persistent_output_buffer=output_buf,  # Optional buffer
            ...     in_dim=2,
            ...     out_dim=3,
            ...     cluster_axis=1
            ... )
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
