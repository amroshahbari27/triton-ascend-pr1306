/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 *
 * L1 Cache Transform — Applies L1 staging transformations based on
 * candidates produced by L1CacheAnalysis.
 *
 * Generalized from the original L1CacheOptPass to handle:
 * - N sequential loops (not just 2): e.g., LayerNorm has 3 passes
 * - Multiple shared GM pointers per function
 * - Fused-add patterns where intermediate is written then re-read
 *
 * Transform strategies:
 * - CACHE_REUSE: Stage GM→L1 on first load, replace subsequent loads with L1→UB
 */

#include "ascend/include/L1CacheOpt/L1CacheAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "l1-cache-transform"

namespace mlir {
namespace triton {

/// Find the memref::CopyOp in a loop that loads from the given GM root.
static memref::CopyOp findGMLoadCopy(scf::ForOp loop, Value gmRoot) {
  memref::CopyOp result;
  loop.walk([&](memref::CopyOp copyOp) {
    if (result)
      return;
    Value root = traceToGMRoot(copyOp.getSource());
    if (root == gmRoot)
      result = copyOp;
  });
  return result;
}

/// Find the UB allocation targeted by a copy op.
static memref::AllocOp findCopyDestAlloc(memref::CopyOp copyOp) {
  Value dst = copyOp.getTarget();
  if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
    if (auto alloc = subview.getSource().getDefiningOp<memref::AllocOp>())
      return alloc;
  }
  if (auto alloc = dst.getDefiningOp<memref::AllocOp>())
    return alloc;
  return nullptr;
}

/// Apply CACHE_REUSE transform for a single L1 candidate.
///
/// For N loops accessing the same GM region:
///   Loop 0 (first): insert ND2NZOp after existing GM→UB copy (stages tile to L1)
///   Loops 1..N-1: replace GM→UB copy with L12UBOp from L1
///   Between loops: insert SetFlag/WaitFlag sync
///
/// This generalizes the original 2-loop pattern to N loops.
bool applyCacheReuseTransform(func::FuncOp funcOp,
                              const L1Candidate &candidate) {
  if (candidate.region.containingLoops.size() < 2) {
    llvm::errs() << "[L1-Transform] Need >= 2 loops for CACHE_REUSE\n";
    return false;
  }

  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  const auto &region = candidate.region;
  int64_t tileElems = 1;
  for (int64_t d : region.shape)
    tileElems *= d;
  Type elemType = region.elementType;
  int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;

  bool is2DTile = (region.shape.size() >= 2 && region.shape[0] > 1);

  // NZ fractal layout depends on whether the tile is 1D or 2D.
  // 1D (POC): flatten to {tileElems/16, 16} → NZ [1, outerRows, 16, 16]
  // 2D: use actual {M, N} → NZ [N/16, ceil(M/16), 16, 16]
  int64_t outerCols, outerRows;
  if (is2DTile) {
    int64_t M = region.shape[0];
    int64_t N = region.shape[1];
    if (N % kNZBlockDim != 0) {
      llvm::errs() << "[L1-Transform] N=" << N
                   << " not divisible by " << kNZBlockDim << ", skipping\n";
      return false;
    }
    outerCols = N / kNZBlockDim;
    outerRows = (M + kNZBlockDim - 1) / kNZBlockDim;
  } else {
    if (tileElems % (kNZBlockDim * kNZBlockDim) != 0) {
      llvm::errs() << "[L1-Transform] tileElems=" << tileElems
                   << " not divisible by " << (kNZBlockDim * kNZBlockDim)
                   << ", skipping\n";
      return false;
    }
    outerCols = 1;
    outerRows = tileElems / (kNZBlockDim * kNZBlockDim);
  }

  int64_t nzElemsPerTile = outerCols * outerRows * kNZBlockDim * kNZBlockDim;
  int64_t totalOuterRows = candidate.numTiles * outerRows;
  int64_t totalL1Bytes =
      candidate.numTiles * nzElemsPerTile * elemBytes;

  if (totalL1Bytes > kL1CapacityBytes) {
    llvm::errs() << "[L1-Transform] L1 footprint " << totalL1Bytes
                 << " exceeds capacity " << kL1CapacityBytes << ", skipping\n";
    return false;
  }

  // === Allocate L1 buffer (NZ fractal format, all tiles for one row) ===
  scf::ForOp firstLoop = region.containingLoops[0];
  builder.setInsertionPoint(firstLoop);
  Location loc = firstLoop.getLoc();

  auto l1AddrSpace =
      hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L1);
  auto l1AllocType = MemRefType::get(
      {outerCols, totalOuterRows, kNZBlockDim, kNZBlockDim}, elemType,
      MemRefLayoutAttrInterface{}, l1AddrSpace);

  auto l1Alloc = builder.create<memref::AllocOp>(loc, l1AllocType);

  auto markOp = builder.create<annotation::MarkOp>(loc, l1Alloc.getResult());
  markOp->setAttr("effects", builder.getArrayAttr(
                                 {builder.getStringAttr("write"),
                                  builder.getStringAttr("read")}));
  markOp->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
                  hivm::HIVMTightlyCoupledBufferAttr::get(ctx, 0));

  llvm::errs() << "[L1-Transform] L1 alloc: " << l1AllocType
               << " (" << totalL1Bytes << " bytes)"
               << (is2DTile ? " [2D tile]" : " [1D tile]") << "\n";

  auto l1TileType = MemRefType::get(
      {outerCols, outerRows, kNZBlockDim, kNZBlockDim}, elemType,
      StridedLayoutAttr::get(
          ctx, ShapedType::kDynamic,
          {totalOuterRows * kNZBlockDim * kNZBlockDim,
           kNZBlockDim * kNZBlockDim, kNZBlockDim, 1}),
      l1AddrSpace);

  auto dstCont = UnitAttr::get(ctx);

  // Helper to compute tile index from loop IV: tileIdx = (iv - lb) / step
  auto computeTileIndex = [&](OpBuilder &b, Location loc,
                              scf::ForOp loop) -> Value {
    Value iv = loop.getInductionVar();
    Value lb = loop.getLowerBound();
    Value step = loop.getStep();
    Value offset = b.create<arith::SubIOp>(loc, iv, lb);
    Value tileIdx = b.create<arith::DivSIOp>(loc, offset, step);
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), tileIdx);
  };

  // Helper to create L1 subview for a given tile index
  auto createL1Subview = [&](OpBuilder &b, Location loc,
                              scf::ForOp loop) -> memref::SubViewOp {
    Value tileIdxIdx = computeTileIndex(b, loc, loop);
    Value outerRowsConst =
        b.create<arith::ConstantIndexOp>(loc, outerRows);
    Value l1Offset = b.create<arith::MulIOp>(loc, tileIdxIdx, outerRowsConst);

    return b.create<memref::SubViewOp>(
        loc, l1TileType, l1Alloc.getResult(),
        SmallVector<OpFoldResult>{b.getIndexAttr(0), l1Offset,
                                  b.getIndexAttr(0), b.getIndexAttr(0)},
        SmallVector<OpFoldResult>{
            b.getIndexAttr(outerCols), b.getIndexAttr(outerRows),
            b.getIndexAttr(kNZBlockDim), b.getIndexAttr(kNZBlockDim)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                  b.getIndexAttr(1), b.getIndexAttr(1)});
  };

  // === Loop 0: Stage each tile GM→L1 via ND2NZOp ===
  {
    memref::CopyOp gmCopy = findGMLoadCopy(firstLoop, region.base);
    if (!gmCopy) {
      llvm::errs() << "[L1-Transform] No GM copy in first loop\n";
      return false;
    }

    builder.setInsertionPointAfter(gmCopy);
    Location loc0 = gmCopy.getLoc();

    // Find the GM source for ND2NZ: walk up from the copy's source
    // to find the reinterpret_cast with static shape and proper strides.
    Value gmSrc = gmCopy.getSource();
    auto gmSrcType = cast<MemRefType>(gmSrc.getType());
    Attribute gmAddrSpace = gmSrcType.getMemorySpace();

    // Trace through subview→reinterpret_cast chain to get the
    // static-shaped GM view with correct strides.
    Value nd2nzSrc;
    if (auto subview = gmSrc.getDefiningOp<memref::SubViewOp>()) {
      nd2nzSrc = subview.getSource();
    } else {
      nd2nzSrc = gmSrc;
    }

    // If the GM source is already rank-2 and static with correct strides,
    // use it directly. Otherwise, create a flat contiguous reinterpretation.
    auto nd2nzSrcType = cast<MemRefType>(nd2nzSrc.getType());
    bool isContiguous1D = (nd2nzSrcType.getRank() == 1) ||
                          (region.shape.size() == 1);

    if (isContiguous1D) {
      // 1D tile (POC pattern): flatten to {flatRows, 16} contiguous
      int64_t flatRows = tileElems / kNZBlockDim;
      auto metadata = builder.create<memref::ExtractStridedMetadataOp>(
          loc0, gmSrc);
      Value dynOffset = metadata.getOffset();

      auto gm2dType = MemRefType::get(
          {flatRows, kNZBlockDim}, elemType,
          StridedLayoutAttr::get(ctx, ShapedType::kDynamic,
                                 {kNZBlockDim, 1}),
          gmAddrSpace);

      nd2nzSrc = builder.create<memref::ReinterpretCastOp>(
          loc0, gm2dType, gmSrc,
          OpFoldResult(dynOffset),
          SmallVector<OpFoldResult>{builder.getIndexAttr(flatRows),
                                    builder.getIndexAttr(kNZBlockDim)},
          SmallVector<OpFoldResult>{builder.getIndexAttr(kNZBlockDim),
                                    builder.getIndexAttr(1)});
    }
    // else: nd2nzSrc is the reinterpret_cast with correct 2D shape+strides

    auto l1Subview = createL1Subview(builder, loc0, firstLoop);

    builder.create<hivm::ND2NZOp>(loc0, TypeRange{},
                                  nd2nzSrc, l1Subview.getResult(),
                                  dstCont);

    llvm::errs() << "[L1-Transform] Inserted ND2NZOp in loop 0"
                 << " (src shape: " << cast<MemRefType>(nd2nzSrc.getType())
                 << ")\n";
  }

  // === Synchronization ===
  // Do NOT hand-insert SetFlag/WaitFlag. BiShengIR's GraphSyncSolver / InjectSync
  // analyzes the ND2NZ (MTE2) → L12UB (MTE1) memory dependency and emits the
  // correct sync automatically (CLAUDE.md rule 10). Manual flags here double up
  // with the solver's and tripped the simulator's
  // "execute_set_flag already has same set_flag" guard.
  llvm::errs() << "[L1-Transform] Deferring MTE2->MTE1 sync to GraphSyncSolver.\n";

  // === For each subsequent loop (1..N-1): replace GM→UB with L12UBOp ===
  for (size_t loopIdx = 1; loopIdx < region.containingLoops.size();
       ++loopIdx) {
    scf::ForOp curLoop = region.containingLoops[loopIdx];

    // Replace GM→UB copy in this loop with L12UBOp
    memref::CopyOp gmCopy = findGMLoadCopy(curLoop, region.base);
    if (!gmCopy) {
      llvm::errs() << "[L1-Transform] No GM copy in loop " << loopIdx
                   << ", skipping\n";
      continue;
    }

    memref::AllocOp ubAlloc = findCopyDestAlloc(gmCopy);
    if (!ubAlloc) {
      llvm::errs() << "[L1-Transform] No UB alloc for loop " << loopIdx
                   << " copy\n";
      continue;
    }

    builder.setInsertionPoint(gmCopy);
    Location loc2 = gmCopy.getLoc();

    auto l1Subview = createL1Subview(builder, loc2, curLoop);

    // New explicit-UB alloc: for 2D tiles, match original tile shape;
    // for 1D tiles, use the NZ-compatible flat 2D shape.
    SmallVector<int64_t> ub2dShape;
    if (is2DTile) {
      for (int64_t d : region.shape)
        ub2dShape.push_back(d);
    } else {
      int64_t nzRows = tileElems / kNZBlockDim;
      ub2dShape = {nzRows, kNZBlockDim};
    }
    auto ubSpaceAttr =
        hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
    auto ubMemrefType =
        MemRefType::get(ub2dShape, elemType, nullptr, ubSpaceAttr);
    auto plainMemrefType = MemRefType::get(ub2dShape, elemType);

    auto newUBAlloc = builder.create<memref::AllocOp>(loc2, ubMemrefType);
    auto plainCast = builder.create<memref::MemorySpaceCastOp>(
        loc2, plainMemrefType, newUBAlloc.getResult());

    // Stride alignment mark
    auto markAlign =
        builder.create<annotation::MarkOp>(loc2, newUBAlloc.getResult());
    markAlign->setAttr(hivm::StrideAlignDimsAttr::name,
                       builder.getDenseI32ArrayAttr({1}));
    markAlign->setAttr(hivm::StrideAlignValueInByteAttr::name,
                       builder.getDenseI32ArrayAttr({32}));

    // L12UBOp: L1 (NZ) → UB (ND)
    builder.create<hivm::L12UBOp>(loc2, TypeRange{},
                                  l1Subview.getResult(),
                                  newUBAlloc.getResult());

    // Create tensor from the plain cast and redirect downstream
    auto tensorType = RankedTensorType::get(ub2dShape, elemType);
    auto toTensor = builder.create<bufferization::ToTensorOp>(
        loc2, tensorType, plainCast.getResult(), true, true);

    // Find and redirect the old ToTensorOp
    Value oldUBResult = ubAlloc.getResult();
    bufferization::ToTensorOp oldToTensor;

    // Direct users
    for (Operation *user : oldUBResult.getUsers()) {
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (tt->getBlock() == gmCopy->getBlock()) {
          oldToTensor = tt;
          break;
        }
      }
    }
    // Through reinterpret_cast/subview chains
    if (!oldToTensor) {
      for (Operation *user : oldUBResult.getUsers()) {
        for (Operation *user2 : user->getResult(0).getUsers()) {
          if (auto tt = dyn_cast<bufferization::ToTensorOp>(user2)) {
            if (tt->getBlock() == gmCopy->getBlock()) {
              oldToTensor = tt;
              break;
            }
          }
          if (oldToTensor)
            break;
        }
        if (oldToTensor)
          break;
      }
    }

    if (oldToTensor) {
      auto oldTensorType =
          cast<RankedTensorType>(oldToTensor.getResult().getType());
      Value reshapedTensor = toTensor.getResult();

      // Compare old and new tensor shapes to determine if reshape is needed
      auto newShape = ArrayRef<int64_t>(ub2dShape);
      auto oldShape = oldTensorType.getShape();

      if (newShape == oldShape) {
        // Shapes match (2D tile where UB matches original): no reshape
      } else if (oldTensorType.getRank() == 1) {
        // 1D original (e.g., tensor<4096xf16>): collapse NZ 2D→1D
        reshapedTensor = builder.create<tensor::CollapseShapeOp>(
            loc2, oldTensorType, reshapedTensor,
            SmallVector<ReassociationIndices>{{0, 1}});
      } else {
        // Different shapes: collapse to 1D then expand to original
        auto flatType = RankedTensorType::get({tileElems}, elemType);
        auto flat = builder.create<tensor::CollapseShapeOp>(
            loc2, flatType, reshapedTensor,
            SmallVector<ReassociationIndices>{{0, 1}});

        SmallVector<ReassociationIndices> expandMap;
        ReassociationIndices allDims;
        for (int64_t i = 0; i < oldTensorType.getRank(); ++i)
          allDims.push_back(i);
        expandMap.push_back(allDims);

        reshapedTensor = builder.create<tensor::ExpandShapeOp>(
            loc2, oldTensorType, flat.getResult(), expandMap);
      }

      oldToTensor.getResult().replaceAllUsesWith(reshapedTensor);
      llvm::errs() << "[L1-Transform] Redirected tensor reads in loop "
                   << loopIdx << "\n";
    }

    gmCopy.erase();
    llvm::errs() << "[L1-Transform] Replaced GM→UB with L12UBOp in loop "
                 << loopIdx << "\n";
  }

  funcOp->setAttr("triton.l1_cache_opt", builder.getUnitAttr());

  llvm::errs() << "[L1-Transform] Complete: "
               << candidate.l1FootprintBytes << " bytes L1, "
               << candidate.reuseCount << " loops, "
               << candidate.gmBytesRemoved << " GM bytes saved per row\n";

  return true;
}

/// Collect operations inside `region` that are in the def-chain of `value`.
/// Returns them in topological order (dependencies before dependents).
static SmallVector<Operation *> collectDefChainInRegion(Value value,
                                                         Region &region) {
  SmallVector<Operation *> result;
  DenseSet<Operation *> visited;

  std::function<void(Value)> collect = [&](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op || !region.isAncestor(op->getParentRegion()) || visited.count(op))
      return;
    visited.insert(op);
    for (Value operand : op->getOperands())
      collect(operand);
    result.push_back(op);
  };
  collect(value);
  return result;
}

/// Prepare an ND2NZ-compatible source from a GM value.
/// For 1D tiles, reshapes to 2D (flatRows x 16) with proper strides.
/// Returns the ready-to-use ND2NZ source.
static Value prepareND2NZSource(OpBuilder &builder, Location loc,
                                Value gmSrc, int64_t tileElems,
                                Type elemType, ArrayRef<int64_t> tileShape,
                                MLIRContext *ctx) {
  auto gmSrcType = cast<MemRefType>(gmSrc.getType());
  Attribute gmAddrSpace = gmSrcType.getMemorySpace();

  Value nd2nzSrc;
  if (auto subview = gmSrc.getDefiningOp<memref::SubViewOp>())
    nd2nzSrc = subview.getSource();
  else
    nd2nzSrc = gmSrc;

  auto nd2nzSrcType = cast<MemRefType>(nd2nzSrc.getType());
  bool isContiguous1D =
      (nd2nzSrcType.getRank() == 1) || (tileShape.size() == 1);

  if (isContiguous1D) {
    int64_t flatRows = tileElems / kNZBlockDim;
    auto metadata =
        builder.create<memref::ExtractStridedMetadataOp>(loc, gmSrc);
    Value dynOffset = metadata.getOffset();

    auto gm2dType = MemRefType::get(
        {flatRows, kNZBlockDim}, elemType,
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {kNZBlockDim, 1}),
        gmAddrSpace);

    nd2nzSrc = builder.create<memref::ReinterpretCastOp>(
        loc, gm2dType, gmSrc, OpFoldResult(dynOffset),
        SmallVector<OpFoldResult>{builder.getIndexAttr(flatRows),
                                  builder.getIndexAttr(kNZBlockDim)},
        SmallVector<OpFoldResult>{builder.getIndexAttr(kNZBlockDim),
                                  builder.getIndexAttr(1)});
  }
  return nd2nzSrc;
}

/// Apply PREFETCH transform for a single-loop L1 candidate.
///
/// Double-buffered software-pipelined pattern (matches AscendC L1 kernel):
///   Allocate L1 with 2 slots (double-buffered):
///     L1[0] = outerRows * 16 * 16 elements, L1[1] = same
///
///   Prologue: ND2NZ GM[first] → L1[slot=0] (MTE2)
///
///   Loop iteration i (tileIdx = (iv-lb)/step, slot = tileIdx%2):
///     1. L12UB: L1[slot] → UB  (MTE1, current tile prefetched previously)
///     2. Compute on UB           (V pipe)
///     3. Store: UB → GM          (MTE3)
///     4. ND2NZ: GM[iv+step] → L1[nextSlot] (MTE2, nextSlot = (tileIdx+1)%2)
///
///   Key: L12UB reads L1[slot], ND2NZ writes L1[nextSlot]. Different slots →
///   no dependency → MTE2 prefetch runs truly async, overlapping with MTE3
///   store. GraphSyncSolver handles the loop-carried MTE2→MTE1 sync.
bool applyPrefetchTransform(func::FuncOp funcOp,
                            const L1Candidate &candidate) {
  if (candidate.region.containingLoops.empty()) {
    llvm::errs() << "[L1-Prefetch] No loops in candidate\n";
    return false;
  }

  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  const auto &region = candidate.region;
  scf::ForOp loop = region.containingLoops[0];

  int64_t tileElems = 1;
  for (int64_t d : region.shape)
    tileElems *= d;
  Type elemType = region.elementType;
  int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;

  bool is2DTile = (region.shape.size() >= 2 && region.shape[0] > 1);

  int64_t outerCols, outerRows;
  if (is2DTile) {
    int64_t M = region.shape[0];
    int64_t N = region.shape[1];
    if (N % kNZBlockDim != 0) {
      llvm::errs() << "[L1-Prefetch] N=" << N << " not NZ-aligned\n";
      return false;
    }
    outerCols = N / kNZBlockDim;
    outerRows = (M + kNZBlockDim - 1) / kNZBlockDim;
  } else {
    if (tileElems % (kNZBlockDim * kNZBlockDim) != 0) {
      llvm::errs() << "[L1-Prefetch] tileElems=" << tileElems
                   << " not NZ-aligned\n";
      return false;
    }
    outerCols = 1;
    outerRows = tileElems / (kNZBlockDim * kNZBlockDim);
  }

  constexpr int64_t NUM_L1_SLOTS = 2;
  int64_t nzElemsPerTile = outerCols * outerRows * kNZBlockDim * kNZBlockDim;
  int64_t l1BytesPerSlot = nzElemsPerTile * elemBytes;
  int64_t totalL1Bytes = l1BytesPerSlot * NUM_L1_SLOTS;
  int64_t totalOuterRows = NUM_L1_SLOTS * outerRows;

  if (totalL1Bytes > kL1CapacityBytes) {
    llvm::errs() << "[L1-Prefetch] Double-buffered L1 footprint "
                 << totalL1Bytes << " exceeds capacity "
                 << kL1CapacityBytes << ", skipping\n";
    return false;
  }

  llvm::errs() << "[L1-Prefetch] tile=" << tileElems << "x" << elemType
               << ", NZ [" << outerCols << "x" << outerRows << "x"
               << kNZBlockDim << "x" << kNZBlockDim << "]"
               << ", L1=" << totalL1Bytes << " bytes (2 slots)"
               << (is2DTile ? " [2D]" : " [1D]") << "\n";

  // === Allocate double-buffered L1 (2 tile slots, NZ format) ===
  auto l1AddrSpace =
      hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L1);
  auto l1AllocType = MemRefType::get(
      {outerCols, totalOuterRows, kNZBlockDim, kNZBlockDim}, elemType,
      MemRefLayoutAttrInterface{}, l1AddrSpace);

  builder.setInsertionPoint(loop);
  Location loc = loop.getLoc();

  auto l1Alloc = builder.create<memref::AllocOp>(loc, l1AllocType);

  auto markOp = builder.create<annotation::MarkOp>(loc, l1Alloc.getResult());
  markOp->setAttr("effects", builder.getArrayAttr(
                                 {builder.getStringAttr("write"),
                                  builder.getStringAttr("read")}));
  markOp->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
                  hivm::HIVMTightlyCoupledBufferAttr::get(ctx, 0));

  llvm::errs() << "[L1-Prefetch] L1 alloc: " << l1AllocType
               << " (double-buffered)\n";

  auto dstCont = UnitAttr::get(ctx);

  // L1 subview type: one tile slot within the double-buffered allocation
  auto l1SlotType = MemRefType::get(
      {outerCols, outerRows, kNZBlockDim, kNZBlockDim}, elemType,
      StridedLayoutAttr::get(
          ctx, ShapedType::kDynamic,
          {totalOuterRows * kNZBlockDim * kNZBlockDim,
           kNZBlockDim * kNZBlockDim, kNZBlockDim, 1}),
      l1AddrSpace);

  // Helper: create L1 subview for a given slot offset (index type)
  auto createL1SlotSubview = [&](OpBuilder &b, Location loc,
                                  Value slotRowOffset) -> memref::SubViewOp {
    return b.create<memref::SubViewOp>(
        loc, l1SlotType, l1Alloc.getResult(),
        SmallVector<OpFoldResult>{b.getIndexAttr(0), slotRowOffset,
                                  b.getIndexAttr(0), b.getIndexAttr(0)},
        SmallVector<OpFoldResult>{
            b.getIndexAttr(outerCols), b.getIndexAttr(outerRows),
            b.getIndexAttr(kNZBlockDim), b.getIndexAttr(kNZBlockDim)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                  b.getIndexAttr(1), b.getIndexAttr(1)});
  };

  // Helper: compute slot row offset from tileIdx
  // slotRowOffset = (tileIdx % NUM_L1_SLOTS) * outerRows
  auto computeSlotOffset = [&](OpBuilder &b, Location loc,
                                Value tileIdx) -> Value {
    Value numSlots = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(tileIdx.getType(), NUM_L1_SLOTS));
    Value slot = b.create<arith::RemSIOp>(loc, tileIdx, numSlots);
    Value slotIdx = b.create<arith::IndexCastOp>(
        loc, b.getIndexType(), slot);
    Value outerRowsConst = b.create<arith::ConstantIndexOp>(loc, outerRows);
    return b.create<arith::MulIOp>(loc, slotIdx, outerRowsConst);
  };

  // Helper: compute raw tileIdx (integer type, before IndexCast)
  auto computeTileIdxRaw = [&](OpBuilder &b, Location loc,
                                Value iv) -> Value {
    Value lb = loop.getLowerBound();
    Value step = loop.getStep();
    Value offset = b.create<arith::SubIOp>(loc, iv, lb);
    return b.create<arith::DivSIOp>(loc, offset, step);
  };

  // Find the GM→UB copy in the loop
  memref::CopyOp gmCopy = findGMLoadCopy(loop, region.base);
  if (!gmCopy) {
    llvm::errs() << "[L1-Prefetch] No GM copy in loop\n";
    return false;
  }

  memref::AllocOp ubAlloc = findCopyDestAlloc(gmCopy);
  if (!ubAlloc) {
    llvm::errs() << "[L1-Prefetch] No UB alloc for copy\n";
    return false;
  }

  Value gmSrc = gmCopy.getSource();
  SmallVector<Operation *> addrChain =
      collectDefChainInRegion(gmSrc, loop.getBodyRegion());

  llvm::errs() << "[L1-Prefetch] Address chain: " << addrChain.size()
               << " ops\n";

  // === PROLOGUE: Prefetch first tile GM → L1[slot=0] ===
  {
    builder.setInsertionPoint(loop);
    IRMapping prologueMap;
    prologueMap.map(loop.getInductionVar(), loop.getLowerBound());

    for (auto *op : addrChain) {
      Operation *cloned = builder.clone(*op, prologueMap);
      for (auto [oldRes, newRes] :
           llvm::zip(op->getResults(), cloned->getResults()))
        prologueMap.map(oldRes, newRes);
    }

    Value prologueSrc = prologueMap.lookupOrDefault(gmSrc);
    Value nd2nzSrc = prepareND2NZSource(builder, loc, prologueSrc, tileElems,
                                        elemType, region.shape, ctx);

    Value slot0Offset = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto l1Slot0 = createL1SlotSubview(builder, loc, slot0Offset);

    builder.create<hivm::ND2NZOp>(loc, TypeRange{}, nd2nzSrc,
                                  l1Slot0.getResult(), dstCont);

    llvm::errs() << "[L1-Prefetch] Prologue: ND2NZOp GM[first] → L1[slot=0]\n";
  }

  // === LOOP BODY: L12UB from L1[slot] (current tile) ===
  {
    builder.setInsertionPoint(gmCopy);
    Location copyLoc = gmCopy.getLoc();

    Value iv = loop.getInductionVar();
    Value tileIdx = computeTileIdxRaw(builder, copyLoc, iv);
    Value curSlotOffset = computeSlotOffset(builder, copyLoc, tileIdx);
    auto l1CurSlot = createL1SlotSubview(builder, copyLoc, curSlotOffset);

    SmallVector<int64_t> ub2dShape;
    if (is2DTile) {
      for (int64_t d : region.shape)
        ub2dShape.push_back(d);
    } else {
      int64_t flatRows = tileElems / kNZBlockDim;
      ub2dShape = {flatRows, kNZBlockDim};
    }

    auto ubSpaceAttr =
        hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
    auto ubMemrefType =
        MemRefType::get(ub2dShape, elemType, nullptr, ubSpaceAttr);
    auto plainMemrefType = MemRefType::get(ub2dShape, elemType);

    auto newUBAlloc = builder.create<memref::AllocOp>(copyLoc, ubMemrefType);
    auto plainCast = builder.create<memref::MemorySpaceCastOp>(
        copyLoc, plainMemrefType, newUBAlloc.getResult());

    auto markAlign =
        builder.create<annotation::MarkOp>(copyLoc, newUBAlloc.getResult());
    markAlign->setAttr(hivm::StrideAlignDimsAttr::name,
                       builder.getDenseI32ArrayAttr({1}));
    markAlign->setAttr(hivm::StrideAlignValueInByteAttr::name,
                       builder.getDenseI32ArrayAttr({32}));

    builder.create<hivm::L12UBOp>(copyLoc, TypeRange{},
                                  l1CurSlot.getResult(),
                                  newUBAlloc.getResult());

    llvm::errs() << "[L1-Prefetch] L12UBOp L1[slot] → UB (current tile)\n";

    auto tensorType = RankedTensorType::get(ub2dShape, elemType);
    auto toTensor = builder.create<bufferization::ToTensorOp>(
        copyLoc, tensorType, plainCast.getResult(), true, true);

    Value oldUBResult = ubAlloc.getResult();
    bufferization::ToTensorOp oldToTensor;

    for (Operation *user : oldUBResult.getUsers()) {
      if (auto tt = dyn_cast<bufferization::ToTensorOp>(user)) {
        if (tt->getBlock() == gmCopy->getBlock()) {
          oldToTensor = tt;
          break;
        }
      }
    }
    if (!oldToTensor) {
      for (Operation *user : oldUBResult.getUsers()) {
        for (Operation *user2 : user->getResult(0).getUsers()) {
          if (auto tt = dyn_cast<bufferization::ToTensorOp>(user2)) {
            if (tt->getBlock() == gmCopy->getBlock()) {
              oldToTensor = tt;
              break;
            }
          }
          if (oldToTensor)
            break;
        }
        if (oldToTensor)
          break;
      }
    }

    if (oldToTensor) {
      auto oldTensorType =
          cast<RankedTensorType>(oldToTensor.getResult().getType());
      Value reshapedTensor = toTensor.getResult();

      auto newShape = ArrayRef<int64_t>(ub2dShape);
      auto oldShape = oldTensorType.getShape();

      if (newShape == oldShape) {
      } else if (oldTensorType.getRank() == 1) {
        reshapedTensor = builder.create<tensor::CollapseShapeOp>(
            copyLoc, oldTensorType, reshapedTensor,
            SmallVector<ReassociationIndices>{{0, 1}});
      } else {
        auto flatType = RankedTensorType::get({tileElems}, elemType);
        auto flat = builder.create<tensor::CollapseShapeOp>(
            copyLoc, flatType, reshapedTensor,
            SmallVector<ReassociationIndices>{{0, 1}});
        SmallVector<ReassociationIndices> expandMap;
        ReassociationIndices allDims;
        for (int64_t i = 0; i < oldTensorType.getRank(); ++i)
          allDims.push_back(i);
        expandMap.push_back(allDims);
        reshapedTensor = builder.create<tensor::ExpandShapeOp>(
            copyLoc, oldTensorType, flat.getResult(), expandMap);
      }

      oldToTensor.getResult().replaceAllUsesWith(reshapedTensor);
      llvm::errs() << "[L1-Prefetch] Redirected tensor reads\n";
    } else {
      llvm::errs() << "[L1-Prefetch] WARNING: No old ToTensorOp found\n";
    }

    gmCopy.erase();
  }

  // === LOOP TAIL: Async prefetch NEXT tile → L1[nextSlot] ===
  // Placed after compute + store. Since nextSlot != curSlot, the ND2NZ
  // writes to a DIFFERENT L1 region than the L12UB reads from.
  // This allows MTE2 (prefetch) to overlap with both V (compute)
  // and MTE3 (store) — truly async, no blocking.
  {
    Block *loopBody = loop.getBody();
    auto *terminator = loopBody->getTerminator();
    builder.setInsertionPoint(terminator);
    Location tailLoc = terminator->getLoc();

    Value iv = loop.getInductionVar();
    Value step = loop.getStep();
    Value nextIV = builder.create<arith::AddIOp>(tailLoc, iv, step);

    // Bounds guard: on the final iteration nextIV == upperBound, so the cloned
    // address chain would compute a GM pointer one tile past the buffer and the
    // ND2NZ prefetch would read out of bounds. Clamp the *address* IV to the last
    // valid iteration (ub - step). The prefetched data lands in nextSlot, which
    // the final iteration never consumes, so re-staging the last tile is harmless.
    Value ub = loop.getUpperBound();
    Value lastValidIV = builder.create<arith::SubIOp>(tailLoc, ub, step);
    Value inBounds = builder.create<arith::CmpIOp>(
        tailLoc, arith::CmpIPredicate::slt, nextIV, ub);
    Value addrIV =
        builder.create<arith::SelectOp>(tailLoc, inBounds, nextIV, lastValidIV);

    Value tileIdx = computeTileIdxRaw(builder, tailLoc, iv);
    Value one = builder.create<arith::ConstantOp>(
        tailLoc, builder.getIntegerAttr(tileIdx.getType(), 1));
    Value nextTileIdx = builder.create<arith::AddIOp>(tailLoc, tileIdx, one);
    Value nextSlotOffset = computeSlotOffset(builder, tailLoc, nextTileIdx);
    auto l1NextSlot = createL1SlotSubview(builder, tailLoc, nextSlotOffset);

    IRMapping nextMap;
    nextMap.map(loop.getInductionVar(), addrIV);

    for (auto *op : addrChain) {
      Operation *cloned = builder.clone(*op, nextMap);
      for (auto [oldRes, newRes] :
           llvm::zip(op->getResults(), cloned->getResults()))
        nextMap.map(oldRes, newRes);
    }

    Value nextSrc = nextMap.lookupOrDefault(gmSrc);
    Value nd2nzSrc = prepareND2NZSource(builder, tailLoc, nextSrc, tileElems,
                                        elemType, region.shape, ctx);

    builder.create<hivm::ND2NZOp>(tailLoc, TypeRange{}, nd2nzSrc,
                                  l1NextSlot.getResult(), dstCont);

    llvm::errs() << "[L1-Prefetch] Loop tail: ND2NZOp GM[next] → L1[nextSlot] "
                 << "(async prefetch, different slot from L12UB)\n";
  }

  funcOp->setAttr("triton.l1_cache_opt", builder.getUnitAttr());

  llvm::errs() << "[L1-Prefetch] Complete: " << totalL1Bytes
               << " bytes L1 (double-buffered), async prefetch\n";

  return true;
}

} // namespace triton
} // namespace mlir
