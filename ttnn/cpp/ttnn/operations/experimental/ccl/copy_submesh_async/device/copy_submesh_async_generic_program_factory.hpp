// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "copy_submesh_async_generic_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <vector>

namespace ttnn::experimental::prim {

struct CopySubmeshAsyncGenericProgram {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle sender_reader_kernel_id;
        tt::tt_metal::KernelHandle sender_writer_kernel_id;
        std::vector<CoreCoord> sender_worker_cores;
        tt::tt_metal::GlobalSemaphore init_barrier_semaphore;
        tt::tt_metal::GlobalSemaphore final_barrier_semaphore;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const CopySubmeshAsyncGenericParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const CopySubmeshAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const CopySubmeshAsyncGenericParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const CopySubmeshAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value,
        const tt::tt_metal::GlobalSemaphore& init_barrier_semaphore,
        const tt::tt_metal::GlobalSemaphore& final_barrier_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const CopySubmeshAsyncGenericParams& operation_attributes,
        const CopySubmeshAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
