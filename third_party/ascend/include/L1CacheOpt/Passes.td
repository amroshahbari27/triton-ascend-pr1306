/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 */

#ifndef TRITON_L1_CACHE_OPT_PASSES
#define TRITON_L1_CACHE_OPT_PASSES

include "mlir/Pass/PassBase.td"

def L1CacheOpt : Pass<"l1-cache-opt", "mlir::ModuleOp"> {
  let summary = "L1 cache staging for multi-pass kernels (e.g. RMSNorm)";
  let description = [{
    Detects sequential scf.for loops that load from the same GM memref and
    transforms the pattern to stage data through L1 (cbuf). The first loop
    copies GM→UB→L1 and the second loop reads L1→UB instead of re-fetching
    from GM. This yields ~17-21% speedup on Ascend 950 for 2-pass patterns.

    The pass sets the kernel to AIC_MIX mode and inserts hivm.copy operations
    with appropriate address spaces and inter-block synchronization.
  }];
  let constructor = "triton::createL1CacheOptPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "annotation::AnnotationDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];
}

#endif // TRITON_L1_CACHE_OPT_PASSES
