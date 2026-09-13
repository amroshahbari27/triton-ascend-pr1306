/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Declares semantics-preserving shaped-vector rewrites used before graph build.
// Scheduling and storage policy are intentionally absent from this file.
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
def CVSplitShapedVF : Constraint<CPred<"::llvm::isa<::mlir::ShapedType>($0.getType())">, "shaped VF expression">;
def CVSplitFoldIdempotentAnd : Pat<(Arith_AndIOp:$root $value, $value),
  (replaceWithValue $value), [(CVSplitShapedVF $value)]>;
def CVSplitFoldIdempotentOr : Pat<(Arith_OrIOp:$root $value, $value),
  (replaceWithValue $value), [(CVSplitShapedVF $value)]>;

def CVSplitUnmarkedRowNormalizeExp : Constraint<
  CPred<"isRowNormalizeExpCandidate($0)">,
  "unmarked static row-normalize exponential">;
def CVSplitUnmarkedOrderedAffine : Constraint<
  CPred<"isOrderedAffineCandidate($0)">,
  "unmarked ordered affine step">;
def CVSplitMarkRowNormalizeExp : NativeCodeCall<
  "cloneWithVFRole($_builder, $0, \"row-normalize-exp\")">;
def CVSplitMarkOrderedAffine : NativeCodeCall<
  "cloneWithVFRole($_builder, $0, \"ordered-affine\")">;

def CVSplitRecognizeRowNormalizeExp : Pat<
  (Math_ExpOp:$root $operand, $fastMath),
  (CVSplitMarkRowNormalizeExp $root),
  [(CVSplitUnmarkedRowNormalizeExp $root)]>;
def CVSplitRecognizeOrderedAffine : Pat<
  (Arith_AddFOp:$root $lhs, $rhs, $fastMath),
  (CVSplitMarkOrderedAffine $root),
  [(CVSplitUnmarkedOrderedAffine $root)]>;
