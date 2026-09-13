/* Copyright (c) Huawei Technologies Co., Ltd. 2025. SPDX-License-Identifier: MIT */
// Registers the conservative CV-split transformation pipeline.
// Its implementation keeps unsupported candidates on the original IR path.
#pragma once
#include "CVSplitScheduling.h"
namespace mlir::triton {
#define GEN_PASS_REGISTRATION
#include "ascend/include/CVSplitScheduling/Passes.h.inc"
} // namespace mlir::triton
