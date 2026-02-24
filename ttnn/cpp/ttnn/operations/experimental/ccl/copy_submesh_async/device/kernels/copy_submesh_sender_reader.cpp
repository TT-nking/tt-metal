// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_tiles_to_read = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t input_address = get_arg_val<address_t>(arg_idx++);
    constexpr auto input_tensor_args = TensorAccessorArgs<3>();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_address, input_page_size);
    for (uint32_t num_tiles_read = 0; num_tiles_read < num_tiles_to_read; num_tiles_read++) {
        cb_reserve_back(cb0_id, 1);
        address_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t noc_read_addr = get_noc_addr(num_tiles_read, input_addrgen);
        // noc_async_read(618477210656, 103712, input_page_size);
        noc_async_read(noc_read_addr, l1_write_addr, input_page_size);
        //  DPRINT << " Read NOC Address = " << noc_read_addr << ", L1 Address = " << l1_write_addr << ENDL();
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
    }
}
