// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ckernel.h"
#include "fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t current_device_id = get_compile_time_arg_val(1);
constexpr int32_t dest_device_id = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_tiles_to_write = get_compile_time_arg_val(4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t global_init_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto output_tensor_args = TensorAccessorArgs<5>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address, output_page_size);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION>(arg_idx);
    if (fabric_connection.is_logically_connected()) {
        DPRINT << "Open Fabric Connection " << ENDL();
        fabric_connection.open();
    }

    DPRINT << "Writer kernel start " << ENDL();
    for (uint32_t num_tiles_written = 0; num_tiles_written < num_tiles_to_write; num_tiles_written++) {
        cb_wait_front(cb0_id, 1);
        address_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc_write_addr = get_noc_addr(num_tiles_written + 1, output_addrgen);
        noc_async_write(l1_read_addr, noc_write_addr, output_page_size);
        // DPRINT << "Write NOC Address = " << noc_write_addr << ", L1 Address = " << l1_read_addr << ENDL();
        noc_async_write_barrier();
        cb_pop_front(cb0_id, 1);
    }
    // noc_async_write_barrier();
    // DPRINT << "finished writer kernel" << ENDL();
    fabric_connection.close();
}
