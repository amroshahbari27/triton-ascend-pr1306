/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier:
 * MIT */
// Run preparation, graph scheduling, and physical lowering as one transaction.
// A rejected candidate leaves the original module untouched for DCVP fallback.
#include "ascend/include/CVSplitScheduling/CVSplitScheduling.h"
#include "ascend/include/CVSplitScheduling/Attributes.h"
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include <utility>
#define DEBUG_TYPE "cv-split-scheduling"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")
using namespace mlir;
using namespace mlir::triton;
namespace {
using namespace cv_split;
enum class TransformStatus { Applied, NotApplicable, Invalid };
LogicalResult transformFunction(func::FuncOp function, scf::ForOp loop) {
  auto mixMode = function->getAttrOfType<StringAttr>("mix_mode");
  if (!mixMode || mixMode.getValue() != "mix") {
    LDBG("input has no verified MIX launch mapping; preserving fallback");
    return failure();
  }
  LDBG("phase BuildSSAGraph");
  auto graph = buildSSAGraph(loop);
  if (failed(graph)) {
    LDBG("BuildSSAGraph rejected " << function.getSymName());
    return failure();
  }
  LDBG("phase ScheduleSSA");
  if (failed(scheduleSSAGraph(*graph))) {
    LDBG("ScheduleSSA rejected " << function.getSymName());
    return failure();
  }
  LDBG("phase SSABufferAndAllocate");
  if (failed(lowerSSAToBufferAllocation(function, *graph))) {
    LDBG("SSABufferAndAllocate rejected " << function.getSymName());
    return failure();
  }
  function->getParentOfType<ModuleOp>()->setAttr(
      "hivm.disable_auto_tile_and_bind_subblock",
      UnitAttr::get(function.getContext()));
  function->getParentOfType<ModuleOp>()->setAttr(
      kPreserveExplicitScheduleAttr, UnitAttr::get(function.getContext()));
  return success();
}
TransformStatus transformModule(ModuleOp module, int factor, bool rewriteVF) {
  SmallVector<func::FuncOp> definitions;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (!function.isDeclaration())
      definitions.push_back(function);
  // The module-wide fallback marker currently limits transformation to
  // one-kernel modules.
  if (definitions.size() != 1) {
    LDBG("expected one defined kernel function, found " << definitions.size());
    return TransformStatus::NotApplicable;
  }
  func::FuncOp function = definitions.front();
  LDBG("phase PrepareSSA");
  auto loop = prepareSSA(function, factor);
  if (failed(loop))
    return TransformStatus::NotApplicable;
  LDBG("phase VFRewrite");
  if (failed(applyVFRewriteStage(*loop, rewriteVF)))
    return TransformStatus::NotApplicable;
  if (failed(transformFunction(function, *loop))) {
    LDBG("graph scheduling or physical lowering rejected "
         << function.getSymName());
    return TransformStatus::NotApplicable;
  }
  removeEngineTypeAttrs(module);
  module->setAttr(kAppliedAttr,
                  Builder(module.getContext()).getI32IntegerAttr(1));
  return failed(verify(module)) ? TransformStatus::Invalid
                                : TransformStatus::Applied;
}
class CVSplitSchedulingPass
    : public ::impl::CVSplitSchedulingBase<CVSplitSchedulingPass> {
public:
  explicit CVSplitSchedulingPass(const CVSplitSchedulingOptions &options)
      : CVSplitSchedulingBase(options) {}
  void runOnOperation() override {
    ModuleOp original = getOperation();
    if (unrollFactor != 2 && unrollFactor != 4 && unrollFactor != 8) {
      original.emitError() << "cv-split unroll factor must be 2, 4, or 8";
      return signalPassFailure();
    }
    if (!compileOn91095)
      return;
    OwningOpRef<ModuleOp> candidate = original.clone();
    TransformStatus status =
        transformModule(*candidate, unrollFactor, enableVFRewrite);
    if (status == TransformStatus::NotApplicable) {
      LDBG("candidate was not applicable; preserving the DCVP fallback input");
      return;
    }
    if (status == TransformStatus::Invalid) {
      original.emitError("cv-split produced invalid IR");
      return signalPassFailure();
    }
    original->setLoc(candidate->getLoc());
    original->setAttrs(candidate->getOperation()->getAttrs());
    original.getBodyRegion().takeBody(candidate->getBodyRegion());
    LDBG("committed verified CV-split transformation");
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCVSplitSchedulingPass(
    const CVSplitSchedulingOptions &options) {
  return std::make_unique<CVSplitSchedulingPass>(options);
}
