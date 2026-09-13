/* Copyright (c) Huawei Technologies Co., Ltd. 2025. SPDX-License-Identifier: MIT */
// Declares the public CV-split pass constructor and generated options.
// Detailed SSA scheduling and physical lowering stay behind this interface.
#pragma once
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#define GEN_PASS_DECL_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"
#define GEN_PASS_DEF_CVSPLITSCHEDULING
#include "ascend/include/CVSplitScheduling/Passes.h.inc"
namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>> createCVSplitSchedulingPass(const CVSplitSchedulingOptions &options = {});
} // namespace mlir::triton
