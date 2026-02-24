// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) {
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = 0.0f;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {}

}  // namespace sfpu
}  // namespace ckernel
