/* Copyright (c) Huawei Technologies Co., Ltd. 2025. SPDX-License-Identifier: MIT */
// Register the single transactional CV-split pass and its semantic controls.
// Scheduling, storage, and synchronization choices are derived from one SSA graph.
include "mlir/Pass/PassBase.td"
def CVSplitScheduling : Pass<"cv_split_scheduling", "mlir::ModuleOp"> {
  let summary = "Unroll innermost loop and split into concurrent Cube/Vector scopes";
  let constructor = "triton::createCVSplitSchedulingPass()";
  let options = [
    Option<"compileOn91095", "compile-on-910-95", "bool", "false", "compile on 910_95">,
    Option<"unrollFactor", "unroll-factor", "int", "2", "loop unroll factor (2, 4, or 8)">,
    Option<"enableVFRewrite", "enable-vf-rewrite", "bool", "false", "apply safe shaped VF rewrites before scheduling">,
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect", "mlir::linalg::LinalgDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect", "mlir::tensor::TensorDialect",
    "mlir::bufferization::BufferizationDialect", "mlir::math::MathDialect",
    "mlir::hivm::HIVMDialect", "mlir::scope::ScopeDialect",
    "mlir::annotation::AnnotationDialect", "mlir::hacc::HACCDialect"
  ];
}
