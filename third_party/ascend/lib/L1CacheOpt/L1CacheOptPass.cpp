/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 *
 * L1 Cache Optimization Pass for Multi-Pass Kernels.
 *
 * Detects two sequential scf.for loops that load from the same GM memref
 * (e.g., RMSNorm: pass1=sum-of-squares, pass2=normalize). Eliminates the
 * redundant GM reload in loop2 by staging the full row in L1 during loop1,
 * then reading from L1 via L12UBOp (MTE1, NZ→ND) in loop2.
 *
 * Strategy (matching AscendC DataCopy/DataCopyL12UBImpl):
 *   1. Allocate L1 buffer for the FULL ROW (all tiles) in NZ fractal format
 *   2. In loop1: ND2NZOp stages each tile GM→L1 at per-tile offset (MTE2)
 *   3. In loop2: L12UBOp reads each tile L1→UB (MTE1), replaces GM→UB copy
 *
 * Synchronization between MTE2 (ND2NZ) and MTE1 (L12UB) is NOT inserted
 * manually. BiShengIR's GraphSyncSolver pass auto-generates the correct
 * forward-backward double-buffering sync pattern (pre-initialized reverse
 * flags, per-iteration wait/set, post-loop drain). Manual SetFlag/WaitFlag
 * here would conflict with the auto-generated ops, causing simulator
 * "execute_set_flag already has same set_flag" errors at runtime.
 *
 * The kernel MUST already have mix_mode="mix" (e.g., via a tl.dot in the
 * Triton source). This pass does NOT inject fake matmuls.
 */

#include "ascend/include/L1CacheOpt/Passes.h"
#include "ascend/include/L1CacheOpt/L1CacheAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#define DEBUG_TYPE "l1-cache-opt"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_L1CACHEOPT
#include "ascend/include/L1CacheOpt/Passes.h.inc"

namespace {

static SmallVector<Value> findGMSourcesInLoop(scf::ForOp loop) {
  SmallVector<Value> sources;
  loop.walk([&](memref::CopyOp copyOp) {
    Value src = copyOp.getSource();
    Value castSource;
    if (auto subviewOp = src.getDefiningOp<memref::SubViewOp>()) {
      Value svSrc = subviewOp.getSource();
      if (auto castOp = svSrc.getDefiningOp<memref::ReinterpretCastOp>()) {
        castSource = castOp.getSource();
      }
    } else if (auto castOp = src.getDefiningOp<memref::ReinterpretCastOp>()) {
      castSource = castOp.getSource();
    }
    if (castSource) {
      if (auto blockArg = dyn_cast<BlockArgument>(castSource)) {
        if (blockArg.getOwner()->isEntryBlock()) {
          if (!llvm::is_contained(sources, castSource))
            sources.push_back(castSource);
        }
      }
    }
  });
  loop.walk([&](memref::ReinterpretCastOp castOp) {
    Value src = castOp.getSource();
    if (auto blockArg = dyn_cast<BlockArgument>(src)) {
      if (blockArg.getOwner()->isEntryBlock()) {
        if (!llvm::is_contained(sources, src))
          sources.push_back(src);
      }
    }
  });
  return sources;
}

static memref::CopyOp findGMLoadCopy(scf::ForOp loop, Value gmSource) {
  memref::CopyOp result;
  loop.walk([&](memref::CopyOp copyOp) {
    if (result)
      return;
    Value src = copyOp.getSource();
    if (auto subviewOp = src.getDefiningOp<memref::SubViewOp>()) {
      Value svSrc = subviewOp.getSource();
      if (auto castOp = svSrc.getDefiningOp<memref::ReinterpretCastOp>()) {
        if (castOp.getSource() == gmSource) {
          result = copyOp;
        }
      }
    } else if (auto castOp = src.getDefiningOp<memref::ReinterpretCastOp>()) {
      if (castOp.getSource() == gmSource) {
        result = copyOp;
      }
    }
  });
  return result;
}

static memref::AllocOp findAllocForCopyDest(memref::CopyOp copyOp) {
  Value dst = copyOp.getTarget();
  if (auto subviewOp = dst.getDefiningOp<memref::SubViewOp>()) {
    if (auto allocOp = subviewOp.getSource().getDefiningOp<memref::AllocOp>())
      return allocOp;
  }
  if (auto allocOp = dst.getDefiningOp<memref::AllocOp>())
    return allocOp;
  return nullptr;
}

static int64_t getConstantTripCount(scf::ForOp loop) {
  auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
  auto st = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (!lb || !ub || !st || st.value() == 0)
    return -1;
  int64_t trips = (ub.value() - lb.value() + st.value() - 1) / st.value();
  return trips > 0 ? trips : -1;
}

struct L1CacheOptPass : public impl::L1CacheOptBase<L1CacheOptPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool transformed = false;

    const char *strategyEnv = std::getenv("TRITON_L1_STRATEGY");
    bool forcePrefetch = strategyEnv && std::string(strategyEnv) == "PREFETCH";

    module.walk([&](func::FuncOp funcOp) {
      funcOp->setAttr("triton.l1_pass_visited",
                      UnitAttr::get(funcOp.getContext()));
    });

    module.walk([&](func::FuncOp funcOp) {
      if (transformed)
        return;

      if (forcePrefetch) {
        // PREFETCH per-loop: apply double-buffered prefetch to each loop
        // independently. Overlaps next-tile GM→L1 with current-tile compute.
        // Better than CACHE_REUSE for compute-heavy kernels (e.g., softmax
        // with exp()) where V-pipe dominates over MTE.
        transformed = applyPrefetchPerLoop(funcOp);
        return;
      }

      // Path A: Original proven logic for >=2 top-level sibling loops (POC)
      transformed = processFunction(funcOp);

      // Path B+C: Generalized analysis (nested sibling reuse + single-loop prefetch)
      if (!transformed) {
        L1CandidateSet candidates = analyzeL1Candidates(funcOp);
        for (const auto &candidate : candidates.candidates) {
          if (!candidate.isProfitable())
            continue;
          if (candidate.strategy == L1Strategy::CACHE_REUSE) {
            if (applyCacheReuseTransform(funcOp, candidate)) {
              transformed = true;
              break;
            }
          } else if (candidate.strategy == L1Strategy::PREFETCH) {
            if (applyPrefetchTransform(funcOp, candidate)) {
              transformed = true;
              break;
            }
          }
        }
      }
    });
    if (transformed) {
      llvm::errs() << "[L1-CACHE] Transformation applied successfully.\n";
    }
  }

  bool applyPrefetchPerLoop(func::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>())
        loops.push_back(forOp);
    });

    llvm::errs() << "[L1-CACHE] PREFETCH per-loop mode: "
                 << loops.size() << " top-level loops.\n";

    bool anyTransformed = false;
    for (auto &loop : loops) {
      SmallVector<Value> gmSources = findGMSourcesInLoop(loop);
      if (gmSources.empty())
        continue;

      for (Value gmSrc : gmSources) {
        memref::CopyOp gmCopy = findGMLoadCopy(loop, gmSrc);
        if (!gmCopy)
          continue;

        memref::AllocOp ubAlloc = findAllocForCopyDest(gmCopy);
        if (!ubAlloc)
          continue;

        MemRefType tileType = ubAlloc.getType();
        if (!tileType.hasStaticShape())
          continue;

        int64_t tileElems = 1;
        for (int64_t d : tileType.getShape())
          tileElems *= d;

        if (tileElems % (kNZBlockDim * kNZBlockDim) != 0)
          continue;

        L1Candidate candidate;
        candidate.region.base = gmSrc;
        candidate.region.shape =
            SmallVector<int64_t>(tileType.getShape());
        candidate.region.elementType = tileType.getElementType();
        candidate.region.computeByteSize();
        candidate.region.containingLoops.push_back(loop);
        candidate.region.loadOps.push_back(gmCopy);
        candidate.region.isReadOnly = true;
        candidate.reuseCount = 1;
        candidate.numTiles = getConstantTripCount(loop);
        candidate.strategy = L1Strategy::PREFETCH;
        candidate.l1FootprintBytes = candidate.region.byteSize;
        candidate.confidenceScore = 0.9f;

        if (candidate.region.byteSize * 2 > kL1CapacityBytes) {
          llvm::errs() << "[L1-CACHE] Tile too large for double-buffered L1, "
                       << "skipping GM arg in loop.\n";
          continue;
        }

        llvm::errs() << "[L1-CACHE] Applying PREFETCH to loop with tile="
                     << tileElems << "x" << tileType.getElementType()
                     << ", numTiles=" << candidate.numTiles << "\n";

        if (applyPrefetchTransform(funcOp, candidate)) {
          anyTransformed = true;
          break;
        }
      }
      if (anyTransformed)
        break;
    }
    return anyTransformed;
  }

  bool processFunction(func::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>())
        loops.push_back(forOp);
    });

    llvm::errs() << "[L1-CACHE] Function '" << funcOp.getName()
                 << "' has " << loops.size() << " top-level loops.\n";

    if (loops.size() < 2) {
      llvm::errs() << "[L1-CACHE] Need >= 2 loops, found " << loops.size()
                   << ".\n";
      return false;
    }

    scf::ForOp loop1 = loops[0];
    scf::ForOp loop2 = loops[1];

    SmallVector<Value> srcs1 = findGMSourcesInLoop(loop1);
    SmallVector<Value> srcs2 = findGMSourcesInLoop(loop2);

    Value sharedGM;
    for (Value s : srcs1) {
      if (llvm::is_contained(srcs2, s)) {
        sharedGM = s;
        break;
      }
    }
    if (!sharedGM) {
      llvm::errs() << "[L1-CACHE] No shared GM source between loops.\n";
      return false;
    }

    memref::CopyOp gmCopy1 = findGMLoadCopy(loop1, sharedGM);
    memref::CopyOp gmCopy2 = findGMLoadCopy(loop2, sharedGM);
    if (!gmCopy1 || !gmCopy2) {
      llvm::errs()
          << "[L1-CACHE] Could not find GM load copies in both loops.\n";
      return false;
    }

    memref::AllocOp ubAlloc1 = findAllocForCopyDest(gmCopy1);
    if (!ubAlloc1) {
      llvm::errs() << "[L1-CACHE] Could not find UB alloc for loop1 copy.\n";
      return false;
    }

    MemRefType tileType = ubAlloc1.getType();
    if (!tileType.hasStaticShape()) {
      llvm::errs() << "[L1-CACHE] UB alloc has dynamic shape, skipping.\n";
      return false;
    }

    int64_t tileSize = 1;
    for (int64_t dim : tileType.getShape())
      tileSize *= dim;
    Type elemType = tileType.getElementType();
    int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;

    constexpr int64_t kNZ = 16;
    if (tileSize % (kNZ * kNZ) != 0) {
      llvm::errs() << "[L1-CACHE] Tile size " << tileSize
                   << " not divisible by NZ block " << (kNZ * kNZ)
                   << ", skipping.\n";
      return false;
    }

    int64_t nzCols = kNZ;
    int64_t nzRows = tileSize / nzCols;
    int64_t outerCols = nzCols / kNZ;   // = 1
    int64_t outerRows = nzRows / kNZ;   // NZ blocks per tile

    int64_t numTiles = getConstantTripCount(loop1);
    if (numTiles < 1) {
      llvm::errs() << "[L1-CACHE] Cannot determine loop trip count, skipping.\n";
      return false;
    }

    int64_t totalOuterRows = numTiles * outerRows;
    int64_t totalL1Bytes = numTiles * tileSize * elemBytes;
    if (totalL1Bytes > kL1CapacityBytes) {
      llvm::errs() << "[L1-CACHE] Full row (" << totalL1Bytes
                   << " bytes) exceeds L1 capacity, skipping.\n";
      return false;
    }

    llvm::errs() << "[L1-CACHE] Pattern matched: tile=" << tileSize << "x"
                 << elemType << ", numTiles=" << numTiles
                 << ", totalL1=" << totalL1Bytes << " bytes"
                 << " -> NZ 4D [" << outerCols << "x" << totalOuterRows
                 << "x" << kNZ << "x" << kNZ << "]\n";

    MLIRContext *ctx = funcOp.getContext();
    OpBuilder builder(ctx);

    // === L1 buffer allocation (NZ fractal format, full row) ===
    auto l1AddrSpace =
        hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L1);
    auto l1AllocType = MemRefType::get(
        {outerCols, totalOuterRows, kNZ, kNZ}, elemType,
        MemRefLayoutAttrInterface{}, l1AddrSpace);

    builder.setInsertionPoint(loop1);
    Location loc = loop1.getLoc();

    auto l1Alloc = builder.create<memref::AllocOp>(loc, l1AllocType);

    auto markOp = builder.create<annotation::MarkOp>(loc, l1Alloc.getResult());
    auto writeAttr = builder.getStringAttr("write");
    auto readAttr = builder.getStringAttr("read");
    markOp->setAttr("effects", builder.getArrayAttr({writeAttr, readAttr}));
    markOp->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
                    hivm::HIVMTightlyCoupledBufferAttr::get(ctx, 0));

    llvm::errs() << "[L1-CACHE] Created full-row L1 alloc: " << l1AllocType << "\n";

    // Per-tile L1 subview type (4D NZ, used in both loops)
    auto l1TileType = MemRefType::get(
        {outerCols, outerRows, kNZ, kNZ}, elemType,
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                               {totalOuterRows * kNZ * kNZ, kNZ * kNZ, kNZ, 1}),
        l1AddrSpace);

    auto dstCont = UnitAttr::get(ctx);

    // === Loop1: stage each tile GM→L1 via ND2NZOp (MTE2) ===
    {
      builder.setInsertionPointAfter(gmCopy1);
      Location loc1 = gmCopy1.getLoc();

      Value gmSrc1 = gmCopy1.getSource();
      auto gmSrc1Type = cast<MemRefType>(gmSrc1.getType());

      auto metadata1 = builder.create<memref::ExtractStridedMetadataOp>(
          loc1, gmSrc1);
      Value dynOffset1 = metadata1.getOffset();

      auto gm2dType = MemRefType::get(
          {nzRows, nzCols}, elemType,
          StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {nzCols, 1}),
          gmSrc1Type.getMemorySpace());

      SmallVector<OpFoldResult> sizes2d = {
          builder.getIndexAttr(nzRows), builder.getIndexAttr(nzCols)};
      SmallVector<OpFoldResult> strides2d = {
          builder.getIndexAttr(nzCols), builder.getIndexAttr(1)};
      auto gmSrc2d = builder.create<memref::ReinterpretCastOp>(
          loc1, gm2dType, gmSrc1,
          OpFoldResult(dynOffset1), sizes2d, strides2d);

      // Tile index = (iv - lower_bound) / step
      Value iv1 = loop1.getInductionVar();
      Value lb1 = loop1.getLowerBound();
      Value step1 = loop1.getStep();
      Value off1 = builder.create<arith::SubIOp>(loc1, iv1, lb1);
      Value tileIdx1 = builder.create<arith::DivSIOp>(loc1, off1, step1);
      Value tileIdx1Idx = builder.create<arith::IndexCastOp>(
          loc1, builder.getIndexType(), tileIdx1);
      Value outerRowsConst = builder.create<arith::ConstantIndexOp>(
          loc1, outerRows);
      Value l1Offset1 = builder.create<arith::MulIOp>(
          loc1, tileIdx1Idx, outerRowsConst);

      auto l1Subview1 = builder.create<memref::SubViewOp>(
          loc1, l1TileType, l1Alloc.getResult(),
          SmallVector<OpFoldResult>{builder.getIndexAttr(0), l1Offset1,
                                   builder.getIndexAttr(0), builder.getIndexAttr(0)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(outerCols),
                                   builder.getIndexAttr(outerRows),
                                   builder.getIndexAttr(kNZ),
                                   builder.getIndexAttr(kNZ)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1), builder.getIndexAttr(1),
                                   builder.getIndexAttr(1), builder.getIndexAttr(1)});

      builder.create<hivm::ND2NZOp>(loc1, TypeRange{},
                                    gmSrc2d.getResult(), l1Subview1.getResult(),
                                    dstCont);

      llvm::errs() << "[L1-CACHE] Inserted per-tile ND2NZOp GM->L1 in loop1.\n";
    }

    // Sync between loops is handled by BiShengIR's InjectSync/GraphSyncSolver
    // pass which correctly generates the forward-backward double-buffering
    // pattern with pre-initialized reverse flags. Manual sync here would
    // conflict with the auto-generated ops.
    llvm::errs() << "[L1-CACHE] Deferring MTE2->MTE1 sync to InjectSync.\n";

    // === Loop2: Replace GM→UB with L12UBOp from L1 (MTE1, NZ→ND) ===
    // Follows InsertL12UBForDebug pattern exactly:
    //   1. Allocate new UB buffer with explicit #hivm.address_space<ub>
    //   2. MemorySpaceCast to plain memref for downstream use
    //   3. Mark stride alignment (required by L12UBOp hardware)
    //   4. L12UBOp from L1 subview to explicit-UB alloc
    //   5. bufferization::ToTensorOp from the plain cast
    //   6. Replace all downstream uses of old UB alloc's tensor
    {
      memref::AllocOp ubAlloc2 = findAllocForCopyDest(gmCopy2);
      if (!ubAlloc2) {
        llvm::errs() << "[L1-CACHE] Could not find UB alloc for loop2 copy, "
                     << "keeping GM->UB.\n";
        funcOp->setAttr("triton.l1_cache_opt", builder.getUnitAttr());
        return true;
      }

      builder.setInsertionPoint(gmCopy2);
      Location loc2 = gmCopy2.getLoc();

      // Tile index = (iv - lower_bound) / step
      Value iv2 = loop2.getInductionVar();
      Value lb2 = loop2.getLowerBound();
      Value step2 = loop2.getStep();
      Value off2 = builder.create<arith::SubIOp>(loc2, iv2, lb2);
      Value tileIdx2 = builder.create<arith::DivSIOp>(loc2, off2, step2);
      Value tileIdx2Idx = builder.create<arith::IndexCastOp>(
          loc2, builder.getIndexType(), tileIdx2);
      Value outerRowsConst2 = builder.create<arith::ConstantIndexOp>(
          loc2, outerRows);
      Value l1Offset2 = builder.create<arith::MulIOp>(
          loc2, tileIdx2Idx, outerRowsConst2);

      auto l1Subview2 = builder.create<memref::SubViewOp>(
          loc2, l1TileType, l1Alloc.getResult(),
          SmallVector<OpFoldResult>{builder.getIndexAttr(0), l1Offset2,
                                   builder.getIndexAttr(0), builder.getIndexAttr(0)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(outerCols),
                                   builder.getIndexAttr(outerRows),
                                   builder.getIndexAttr(kNZ),
                                   builder.getIndexAttr(kNZ)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(1), builder.getIndexAttr(1),
                                   builder.getIndexAttr(1), builder.getIndexAttr(1)});

      // New UB alloc with EXPLICIT UB address space (InsertL12UBForDebug pattern)
      SmallVector<int64_t> ub2dShape = {nzRows, nzCols};
      auto ubSpaceAttr =
          hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
      auto ubMemrefType = MemRefType::get(ub2dShape, elemType,
                                          /*layout=*/nullptr, ubSpaceAttr);
      auto plainMemrefType = MemRefType::get(ub2dShape, elemType);

      auto newUBAlloc = builder.create<memref::AllocOp>(loc2, ubMemrefType);
      auto plainCast = builder.create<memref::MemorySpaceCastOp>(
          loc2, plainMemrefType, newUBAlloc.getResult());

      // Mark UB alloc stride alignment (required by L12UBOp hardware)
      {
        auto markAlignOp = builder.create<annotation::MarkOp>(
            loc2, newUBAlloc.getResult());
        SmallVector<int32_t> alignDims = {1};   // last dim for rank 2
        SmallVector<int32_t> alignBytes = {32};
        markAlignOp->setAttr(hivm::StrideAlignDimsAttr::name,
                             builder.getDenseI32ArrayAttr(alignDims));
        markAlignOp->setAttr(hivm::StrideAlignValueInByteAttr::name,
                             builder.getDenseI32ArrayAttr(alignBytes));
      }

      // L12UBOp: L1 (NZ) -> UB (ND)
      builder.create<hivm::L12UBOp>(loc2, TypeRange{},
                                    l1Subview2.getResult(),
                                    newUBAlloc.getResult());

      // Create tensor from the plain (no-address-space) cast
      auto tensorType = RankedTensorType::get(ub2dShape, elemType);
      auto toTensor = builder.create<bufferization::ToTensorOp>(
          loc2, tensorType, plainCast.getResult(),
          /*restrict=*/true, /*writable=*/true);

      // Find the existing bufferization::ToTensorOp that reads from old UB
      // and redirect its users to our new tensor (flattened back to 1D)
      Value oldUBResult = ubAlloc2.getResult();
      bufferization::ToTensorOp oldToTensor;
      for (Operation *user : oldUBResult.getUsers()) {
        if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
          if (tt->getBlock() == gmCopy2->getBlock()) {
            oldToTensor = tt;
            break;
          }
        }
      }
      // Also check through reinterpret_cast/subview chains
      if (!oldToTensor) {
        for (Operation *user : oldUBResult.getUsers()) {
          for (Operation *user2 : user->getResult(0).getUsers()) {
            if (auto tt = dyn_cast<bufferization::ToTensorOp>(user2)) {
              if (tt->getBlock() == gmCopy2->getBlock()) {
                oldToTensor = tt;
                break;
              }
            }
          }
          if (oldToTensor) break;
        }
      }

      if (oldToTensor) {
        // Old tensor is 1D (e.g., tensor<4096xf16>), new is 2D.
        // Collapse 2D→1D to match downstream expectations.
        auto old1dTensorType = oldToTensor.getResult().getType();
        auto collapse = builder.create<tensor::CollapseShapeOp>(
            loc2, old1dTensorType, toTensor.getResult(),
            SmallVector<ReassociationIndices>{{0, 1}});
        oldToTensor.getResult().replaceAllUsesWith(collapse.getResult());
        llvm::errs() << "[L1-CACHE] Redirected downstream tensor reads.\n";
      } else {
        llvm::errs() << "[L1-CACHE] WARNING: Could not find old ToTensorOp.\n";
      }

      gmCopy2.erase();

      llvm::errs() << "[L1-CACHE] Replaced loop2 GM->UB with L12UBOp (MTE1) "
                   << "using explicit UB alloc.\n";
    }

    funcOp->setAttr("triton.l1_cache_opt", builder.getUnitAttr());

    llvm::errs() << "[L1-CACHE] Complete: " << totalL1Bytes
                 << " bytes L1 (" << numTiles << " tiles), "
                 << "ND2NZ(MTE2) + sync + L12UB(MTE1).\n";

    return true;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createL1CacheOptPass() {
  return std::make_unique<L1CacheOptPass>();
}

} // namespace triton
} // namespace mlir
