// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_submesh_async_generic_program_factory.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include <cstdint>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <unordered_map>

namespace ttnn::experimental::prim {

namespace {
ttnn::Shape __attribute__((unused)) get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    return ttnn::Shape(tiled_shape);
}
}  // namespace

CopySubmeshAsyncGenericProgram::cached_mesh_workload_t CopySubmeshAsyncGenericProgram::create_mesh_workload(
    const CopySubmeshAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const CopySubmeshAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    const auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id;
    auto subdevice = subdevice_id.has_value() ? *subdevice_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore((IDevice*)mesh_device, available_cores, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore((IDevice*)mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize((MeshDevice*)mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<CopySubmeshAsyncGenericProgram::shared_variables_t>
CopySubmeshAsyncGenericProgram::create_at(
    const CopySubmeshAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const CopySubmeshAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& init_barrier_semaphore,
    const tt::tt_metal::GlobalSemaphore& final_barrier_semaphore) {
    log_debug(tt::LogOp, "DEBUG: create_at is called");
    const auto cluster_axis = operation_attributes.cluster_axis;
    // uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
    //     tensor_args.input_tensor, mesh_coordinate, std::nullopt);

    const auto* full_mesh_device = operation_attributes.union_in_out_mesh_device;
    const auto* this_device = full_mesh_device->get_device(mesh_coordinate);

    std::vector<distributed::MeshCoordinate> full_mesh_device_coords;
    std::map<uint32_t, MeshCoordinate> device_index_to_coord;
    full_mesh_device_coords.reserve(full_mesh_device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(full_mesh_device->shape())) {
        full_mesh_device_coords.push_back(coord);
        // auto this_coord = MeshCoordinate(coord);
        device_index_to_coord.emplace(full_mesh_device->get_device(coord)->id(), coord);
    }

    const auto* dest_mesh_device = operation_attributes.output_mesh_device->get_device(mesh_coordinate);
    const MeshCoordinate& dest_mesh_coord = device_index_to_coord.at(dest_mesh_device->id());

    std::vector<distributed::MeshCoordinate> output_mesh_device_coords;
    for (const auto& coord : distributed::MeshCoordinateRange(operation_attributes.output_mesh_device->shape())) {
        output_mesh_device_coords.push_back(coord);
        log_trace(
            tt::LogOp,
            "Output Mesh Id = {} coord = {}",
            operation_attributes.output_mesh_device->get_device(coord)->id(),
            coord);
    }

    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        full_mesh_device, full_mesh_device_coords, mesh_coordinate, 1, operation_attributes.topology, cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        full_mesh_device, full_mesh_device_coords, mesh_coordinate, -1, operation_attributes.topology, cluster_axis);

    // TT_FATAL(device_index < operation_attributes.num_devices, "DEBUG: device_index: {}", device_index);
    (void)tensor_return_value;

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    uint32_t tile_size_bytes = tt::tile_size(input_data_format);
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    TT_FATAL(
        packet_size_bytes >= tile_size_bytes,
        "Packet size {} must be greater than or equal to tile size {}",
        packet_size_bytes,
        tile_size_bytes);

    log_info(
        tt::LogOp,
        "Creating program for mesh coordinate {} on device {} to dest {},{} forward neighbor {}@{}, backward neighbor "
        "{}@{}",
        mesh_coordinate,
        this_device->id(),
        dest_mesh_device->id(),
        dest_mesh_coord,
        forward_coord,
        forward_coord.has_value() ? full_mesh_device->get_device(forward_coord.value())->id() : -1,
        backward_coord,
        backward_coord.has_value() ? full_mesh_device->get_device(backward_coord.value())->id() : -1);
    tt::tt_metal::Program program{};
    MeshDevice* device = tensor_args.input_tensor.device();
    int32_t forward_id = forward_coord.has_value() ? full_mesh_device->get_device(forward_coord.value())->id() : -1;
    int32_t backward_id = backward_coord.has_value() ? full_mesh_device->get_device(backward_coord.value())->id() : -1;

    log_info(
        tt::LogOp,
        "Creating program for mesh coordinate {} on device {} to dest {},{} forward neighbor {}@{}, backward neighbor "
        "{}@{}",
        mesh_coordinate,
        this_device->id(),
        dest_mesh_device->id(),
        dest_mesh_coord,
        forward_coord,
        forward_id,
        backward_coord,
        backward_id);

    bool send_forward = dest_mesh_coord[cluster_axis] > mesh_coordinate[cluster_axis];
    for (int coord_dim = 0; coord_dim < mesh_coordinate.dims(); coord_dim++) {
        if (coord_dim != cluster_axis) {
            TT_FATAL(
                mesh_coordinate[coord_dim] == dest_mesh_coord[coord_dim],
                "Non-cluster-axis coordinates must match between source and destination at dim {}. Got source "
                "coordinate {}, destination coordinate {}, cluster axis {}",
                coord_dim,
                mesh_coordinate,
                dest_mesh_coord,
                cluster_axis);
        }
    }

    const uint32_t num_senders_per_link = 1;
    const uint32_t num_links = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        ttnn::ccl::choose_worker_cores(num_links, num_senders_per_link, device, operation_attributes.subdevice_id);

    (void)send_forward;

    log_info(
        tt::LogOp,
        "Chosen worker cores for mesh coordinate {}: {}, {}",
        mesh_coordinate,
        sender_worker_cores,
        sender_worker_core_range);

    (void)this_device;
    (void)dest_mesh_device;

    uint32_t temp_CB_index = tt::CB::c_in0;
    const uint32_t num_buffers = 4;
    tt::tt_metal::create_cb(
        temp_CB_index, program, sender_worker_core_range, tile_size_bytes, num_buffers, input_data_format);

    uint32_t num_tiles = tensor_args.input_tensor.buffer()->num_pages();
    log_info(tt::LogOp, "Input tensor has {} tiles", num_tiles);

    std::vector<uint32_t> reader_compile_time_args = {
        temp_CB_index,
        tile_size_bytes,
        num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        tt::CB::c_in0, mesh_coordinate[cluster_axis], dest_mesh_coord[cluster_axis], tile_size_bytes, num_tiles};
    log_info(
        tt::LogOp,
        "Output Buffer = {}, coords = {}",
        tensor_return_value.buffer()->address(),
        tensor_return_value.device_storage().coords);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/copy_submesh_async/device/kernels/copy_submesh_sender_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/copy_submesh_async/device/kernels/copy_submesh_sender_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    log_info(tt::LogOp, "Writer compile args = {}", writer_compile_time_args);
    log_info(tt::LogOp, "Reader compile args = {}", reader_compile_time_args);

    for (auto core : sender_worker_cores) {
        std::vector<uint32_t> reader_rt_args = {tensor_args.input_tensor.buffer()->address()};
        std::vector<uint32_t> writer_rt_args = {
            tensor_return_value.buffer()->address(),
            init_barrier_semaphore.address(),
            final_barrier_semaphore.address()};
        // Used by  FabricConnectionManager::build_from_args to make the connection.
        writer_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
            const auto forward_device_fabric_node_id = full_mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device_fabric_node_id, forward_device_fabric_node_id, 0, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
            const auto backward_device_fabric_node_id = full_mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device_fabric_node_id, backward_device_fabric_node_id, 0, program, {core}, writer_rt_args);
        }
        log_info(
            tt::LogOp, "Runtime args for core {} : reader = {}, writer = {}", core, reader_rt_args, writer_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);
    }
    return {
        std::move(program),
        {.init_barrier_semaphore = init_barrier_semaphore, .final_barrier_semaphore = final_barrier_semaphore}};
}

void CopySubmeshAsyncGenericProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const CopySubmeshAsyncGenericParams& /*operation_attributes*/,
    const CopySubmeshAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = coordinate_range.start_coord();
        TT_FATAL(
            coord == coordinate_range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            coordinate_range.end_coord());
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        auto& sender_reader_runtime_args = GetRuntimeArgs(program, shared_variables.sender_reader_kernel_id);
        auto& sender_writer_runtime_args = GetRuntimeArgs(program, shared_variables.sender_writer_kernel_id);
        for (const auto& core : shared_variables.sender_worker_cores) {
            auto& worker_sender_reader_runtime_args = sender_reader_runtime_args[core.x][core.y];
            auto& worker_sender_writer_runtime_args = sender_writer_runtime_args[core.x][core.y];
            worker_sender_reader_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            worker_sender_writer_runtime_args[0] = tensor_return_value.buffer()->address();
            worker_sender_writer_runtime_args[1] = shared_variables.init_barrier_semaphore.address();
            worker_sender_writer_runtime_args[2] = shared_variables.final_barrier_semaphore.address();
        }
    }
}

}  // namespace ttnn::experimental::prim
