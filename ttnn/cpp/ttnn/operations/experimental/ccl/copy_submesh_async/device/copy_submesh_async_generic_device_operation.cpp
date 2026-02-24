// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_submesh_async_generic_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

void CopySubmeshAsyncGenericDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_size = input_tensor.buffer()->page_size();

    TT_FATAL(page_size % input_tensor.buffer()->alignment() == 0, "CopySubmeshAsync currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to copy_submesh_async must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to copy_submesh_async must be allocated in buffers on device");

    TT_FATAL(input_tensor.layout() == Layout::TILE, "Unsupported input layout {}.", input_tensor.layout());
}

void CopySubmeshAsyncGenericDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    (void)operation_attributes;
    (void)tensor_args;
}

CopySubmeshAsyncGenericDeviceOperation::spec_return_value_t
CopySubmeshAsyncGenericDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    (void)operation_attributes;
    return tensor_args.input_tensor.tensor_spec();
}

CopySubmeshAsyncGenericDeviceOperation::tensor_return_value_t
CopySubmeshAsyncGenericDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args),
        (IDevice*)operation_attributes.union_in_out_mesh_device);
}

tt::stl::hash::hash_t CopySubmeshAsyncGenericDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "CopySubmeshAsyncGenericDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.subdevice_id;
    auto* input_mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(input_mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set =
        input_mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    return tt::tt_metal::operation::hash_operation<CopySubmeshAsyncGenericDeviceOperation>(
        operation_attributes.num_links, operation_attributes.topology, subdevice_core_range_set, tensor_args);
}

Tensor copy_submesh_async_generic(
    const ttnn::Tensor& input_tensor,
    const MeshDevice* output_mesh_device,
    const MeshDevice* union_in_out_mesh_device,
    const uint32_t cluster_axis,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    using OperationType = CopySubmeshAsyncGenericDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mesh_device = output_mesh_device,
        .union_in_out_mesh_device = union_in_out_mesh_device,
        .cluster_axis = cluster_axis,
        .num_links = num_links,
        .topology = topology,
        .subdevice_id = subdevice_id,
    };
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor copy_submesh_async_generic(
    const ttnn::Tensor& input_tensor,
    const MeshDevice* output_mesh_device,
    const MeshDevice* union_in_out_mesh_device,
    const uint32_t cluster_axis,
    uint32_t num_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::experimental::prim::copy_submesh_async_generic(
        input_tensor, output_mesh_device, union_in_out_mesh_device, cluster_axis, num_links, topology, subdevice_id);
}

}  // namespace ttnn::prim
