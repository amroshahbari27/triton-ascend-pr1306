/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Conservatively select and unroll one static SSA loop.
// Reject effects, recurrences, and shapes outside the supported graph subset.
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#define DEBUG_TYPE "cv-split-prepare-ssa"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")
using namespace mlir;
namespace mlir::triton::cv_split {
namespace {
bool hasDynamicShape(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && (!shaped.hasRank() || !shaped.hasStaticShape());
}
bool tracesToEntryArgument(Value value) {
  while (true) {
    if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp())) {
      value = view.getViewSource();
      continue;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument) return false;
    Operation *owner = argument.getOwner()->getParentOp();
    if (isa<func::FuncOp>(owner)) return true;
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || argument.getArgNumber() == 0) return false;
    value = loop.getInitArgs()[argument.getArgNumber() - 1];
  }
}
bool isFreshLoadBuffer(Value buffer, scf::ForOp loop) {
  auto alloc = buffer.getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != loop.getBody() ||
      !alloc.getDynamicSizes().empty()) return false;
  memref::CopyOp writer;
  SmallVector<bufferization::ToTensorOp> readers;
  for (OpOperand &use : buffer.getUses()) {
    if (auto copy = dyn_cast<memref::CopyOp>(use.getOwner());
        copy && copy.getTarget() == buffer) {
      if (writer || copy->getBlock() != loop.getBody() ||
          copy.getSource() == buffer) return false;
      writer = copy;
    } else if (auto tensor = dyn_cast<bufferization::ToTensorOp>(use.getOwner());
        tensor && tensor.getBuffer() == buffer) {
      if (tensor->getBlock() != loop.getBody()) return false;
      readers.push_back(tensor);
    } else return false;
  }
  return writer && tracesToEntryArgument(writer.getSource()) &&
         !readers.empty() && llvm::all_of(readers, [&](auto reader) {
           return writer->isBeforeInBlock(reader); });
}
bool isLoadMaterialization(Operation *op, scf::ForOp loop) {
  if (auto alloc = dyn_cast<memref::AllocOp>(op))
    return isFreshLoadBuffer(alloc.getMemref(), loop);
  auto copy = dyn_cast<memref::CopyOp>(op);
  return copy && isFreshLoadBuffer(copy.getTarget(), loop);
}
bool hasUnsupportedEffect(Operation *op, scf::ForOp loop) {
  if (isLoadMaterialization(op, loop) || isMemoryEffectFree(op)) return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects) return true;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [](const auto &effect) {
    return !isa<MemoryEffects::Read>(effect.getEffect()); });
}
bool hasUnsupportedDynamicOperand(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (!hasDynamicShape(type)) continue;
    auto view = dyn_cast<memref::ReinterpretCastOp>(op);
    if (!view || view.getSource().getType() != type ||
        hasDynamicShape(view.getType())) return true;
  }
  return false;
}
bool hasUnsupportedBody(scf::ForOp loop) {
  if (llvm::any_of(loop.getBody()->getArguments(), [](Value value) {
        return hasDynamicShape(value.getType()); }) ||
      llvm::any_of(loop->getOperandTypes(), hasDynamicShape) ||
      llvm::any_of(loop->getResultTypes(), hasDynamicShape))
    return true;
  return loop.getBody()->walk([loop](Operation *op) {
        bool unsupported =
            isa<RegionBranchOpInterface>(op) || op->getNumSuccessors() ||
            hasUnsupportedDynamicOperand(op) ||
            llvm::any_of(op->getResultTypes(), hasDynamicShape) ||
            hasUnsupportedEffect(op, loop);
        if (unsupported) LDBG("rejecting unsupported operation " << op->getName());
        return unsupported ? WalkResult::interrupt() : WalkResult::advance();
      }).wasInterrupted();
}
FailureOr<int64_t> getTripCount(scf::ForOp loop) {
  auto lower = getConstantIntValue(loop.getLowerBound());
  auto upper = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  int64_t distance;
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
      llvm::SubOverflow(*upper, *lower, distance))
    return failure();
  return distance / *step + (distance % *step != 0);
}
FailureOr<scf::ForOp> findCandidateLoop(func::FuncOp function, int factor) {
  SmallVector<scf::ForOp> candidates;
  function.walk([&](scf::ForOp loop) {
    bool nested = loop.getBody()->walk([](scf::ForOp) {
      return WalkResult::interrupt(); }).wasInterrupted();
    if (!nested) candidates.push_back(loop);
  });
  if (candidates.size() != 1) {
    LDBG("expected one innermost loop, found " << candidates.size());
    return failure();
  }
  scf::ForOp loop = candidates.front();
  auto trips = getTripCount(loop);
  // A fully consumed loop is promoted by A5 unrolling and no longer provides our graph node.
  if (failed(trips) || *trips <= factor || *trips % factor != 0 ||
      hasUnsupportedBody(loop)) {
    LDBG("candidate failed static trip-count, shape, control-flow, or effect "
         "checks");
    return failure();
  }
  return loop;
}
} // namespace
FailureOr<scf::ForOp> prepareSSA(func::FuncOp function, int factor) {
  if (factor != 2 && factor != 4 && factor != 8) return failure();
  auto loop = findCandidateLoop(function, factor);
  if (failed(loop)) return failure();
  Builder builder(function.getContext());
  int64_t origin = 0;
  for (Operation &operation : (*loop).getBody()->without_terminator())
    operation.setAttr(kOriginAttr, builder.getI64IntegerAttr(origin++));
  auto unrolled = loopUnrollByFactor(*loop, factor);
  if (failed(unrolled) || unrolled->epilogueLoopOp || !unrolled->mainLoopOp ||
      failed(verify(function))) {
    LDBG("standard SCF unrolling failed or produced an unexpected remainder");
    return failure();
  }
  scf::ForOp preparedLoop = *unrolled->mainLoopOp;
  LDBG("prepared loop with unroll factor " << factor);
  return preparedLoop;
}
} // namespace mlir::triton::cv_split
