/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 */

#ifndef TRITON_ADAPTER_L1_CACHE_OPT_PASSES_H
#define TRITON_ADAPTER_L1_CACHE_OPT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createL1CacheOptPass();

#define GEN_PASS_REGISTRATION
#include "ascend/include/L1CacheOpt/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_L1_CACHE_OPT_PASSES_H
