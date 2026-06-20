/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 *
 * L1 Cache Analysis — Detects L1 caching opportunities in multi-pass kernels.
 *
 * Algorithm:
 *   1. Collect all top-level scf::ForOp loops in the function.
 *   2. For each loop, find GM sources via memref::CopyOp chains.
 *   3. Group GM sources that appear in multiple loops (shared pointers).
 *   4. For each shared group:
 *      a. Verify region identity (same tile shape, same offset pattern).
 *      b. Check read-only (no stores to same base between loops).
 *      c. Check NZ compatibility (tile divisible by 16×16).
 *      d. Check L1 capacity (total bytes ≤ 512 KB).
 *      e. Score the candidate.
 *   5. Emit L1CandidateSet.
 */

#include "ascend/include/L1CacheOpt/L1CacheAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "l1-cache-analysis"

namespace mlir {
namespace triton {

// --- MemoryRegion implementation ---

bool MemoryRegion::isNZCompatible() const {
  if (shape.empty())
    return false;
  int64_t totalElems = 1;
  for (int64_t d : shape)
    totalElems *= d;
  return totalElems % (kNZBlockDim * kNZBlockDim) == 0;
}

void MemoryRegion::computeByteSize() {
  int64_t totalElems = 1;
  for (int64_t d : shape)
    totalElems *= d;
  int64_t elemBytes = 0;
  if (elementType.isIntOrFloat())
    elemBytes = elementType.getIntOrFloatBitWidth() / 8;
  byteSize = totalElems * elemBytes;
}

bool MemoryRegion::isSameRegion(const MemoryRegion &r1,
                                const MemoryRegion &r2) {
  if (r1.base != r2.base)
    return false;
  if (r1.shape.size() != r2.shape.size())
    return false;
  for (size_t i = 0; i < r1.shape.size(); ++i) {
    if (r1.shape[i] != r2.shape[i])
      return false;
  }
  if (r1.elementType != r2.elementType)
    return false;
  return true;
}

// --- Helper functions ---

Value traceToGMRoot(Value memrefVal) {
  Value current = memrefVal;
  for (int depth = 0; depth < 20; ++depth) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (blockArg.getOwner()->isEntryBlock())
        return current;
    }
    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return Value();

    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      current = subview.getSource();
    } else if (auto cast = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      current = cast.getSource();
    } else if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      current = cast.getSource();
    } else if (auto spaceCast = dyn_cast<memref::MemorySpaceCastOp>(defOp)) {
      current = spaceCast.getSource();
    } else {
      return Value();
    }
  }
  return Value();
}

int64_t getConstantTripCount(scf::ForOp loop) {
  auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
  auto st = loop.getStep().getDefiningOp<arith::ConstantIntOp>();
  if (!lb || !ub || !st || st.value() == 0)
    return -1;
  int64_t trips = (ub.value() - lb.value() + st.value() - 1) / st.value();
  return trips > 0 ? trips : -1;
}

// --- Internal analysis helpers ---

namespace {

/// Information about a GM load within a loop.
struct LoopGMLoad {
  scf::ForOp loop;
  int loopIndex;
  memref::CopyOp copyOp;
  Value gmRoot;        // Traced back to function argument
  int64_t gmArgIndex;  // Function argument index
  memref::AllocOp ubAlloc; // UB allocation destination
};

/// Collect all GM loads from all top-level loops.
SmallVector<LoopGMLoad>
collectGMLoads(ArrayRef<scf::ForOp> loops) {
  SmallVector<LoopGMLoad> loads;
  for (int i = 0, e = loops.size(); i < e; ++i) {
    scf::ForOp loop = loops[i];
    loop.walk([&](memref::CopyOp copyOp) {
      Value src = copyOp.getSource();
      Value gmRoot = traceToGMRoot(src);
      if (!gmRoot)
        return;

      auto blockArg = dyn_cast<BlockArgument>(gmRoot);
      if (!blockArg)
        return;

      LoopGMLoad load;
      load.loop = loop;
      load.loopIndex = i;
      load.copyOp = copyOp;
      load.gmRoot = gmRoot;
      load.gmArgIndex = blockArg.getArgNumber();

      // Find UB alloc destination
      Value dst = copyOp.getTarget();
      if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
        if (auto alloc = subview.getSource().getDefiningOp<memref::AllocOp>())
          load.ubAlloc = alloc;
      } else if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
        load.ubAlloc = alloc;
      }

      loads.push_back(load);
    });
  }
  return loads;
}

/// Group loads by their GM root argument.
using LoadGroup = SmallVector<LoopGMLoad>;
DenseMap<int64_t, LoadGroup>
groupLoadsByGMArg(ArrayRef<LoopGMLoad> loads) {
  DenseMap<int64_t, LoadGroup> groups;
  for (auto &load : loads) {
    groups[load.gmArgIndex].push_back(load);
  }
  return groups;
}

/// Check whether any store exists to the given GM root between loops.
bool hasStoresBetweenLoops(func::FuncOp funcOp, Value gmRoot,
                           scf::ForOp loop1, scf::ForOp loop2) {
  bool betweenLoops = false;
  bool foundStore = false;
  funcOp.walk([&](Operation *op) {
    if (op == loop1.getOperation()) {
      betweenLoops = true;
      return;
    }
    if (op == loop2.getOperation()) {
      betweenLoops = false;
      return;
    }
    if (!betweenLoops)
      return;
    if (auto storeOp = dyn_cast<memref::CopyOp>(op)) {
      Value dst = traceToGMRoot(storeOp.getTarget());
      if (dst == gmRoot)
        foundStore = true;
    }
  });
  return foundStore;
}

/// Extract tile shape from a UB allocation.
SmallVector<int64_t> getTileShape(memref::AllocOp alloc) {
  if (!alloc)
    return {};
  MemRefType type = alloc.getType();
  if (!type.hasStaticShape())
    return {};
  return SmallVector<int64_t>(type.getShape());
}

/// Count distinct loops that load this GM argument.
int countDistinctLoops(const LoadGroup &group) {
  DenseSet<Operation *> seen;
  for (const auto &load : group) {
    scf::ForOp loop = load.loop;
    seen.insert(loop.getOperation());
  }
  return seen.size();
}

/// Build an L1Candidate from a load group.
L1Candidate buildCandidate(func::FuncOp funcOp, int64_t gmArgIndex,
                           const LoadGroup &group,
                           ArrayRef<scf::ForOp> allLoops) {
  L1Candidate candidate;
  candidate.region.base = group[0].gmRoot;
  candidate.region.baseArgIndex = gmArgIndex;

  // Tile shape from first load's UB alloc
  memref::AllocOp firstAlloc = group[0].ubAlloc;
  if (firstAlloc) {
    candidate.region.shape = getTileShape(firstAlloc);
    candidate.region.elementType = firstAlloc.getType().getElementType();
    candidate.region.computeByteSize();
  }

  // Collect containing loops
  DenseSet<Operation *> seenLoops;
  for (const auto &load : group) {
    scf::ForOp loop = load.loop;
    if (seenLoops.insert(loop.getOperation()).second) {
      candidate.region.containingLoops.push_back(loop);
      candidate.region.loadOps.push_back(load.copyOp);
    }
  }

  candidate.reuseCount = candidate.region.containingLoops.size();

  // Trip count from first loop (assume all loops have same trip count)
  if (!allLoops.empty()) {
    candidate.numTiles = getConstantTripCount(allLoops[0]);
  }

  // --- Checks ---

  // Check 1: Need at least 2 loads to benefit from caching
  if (candidate.reuseCount < 2) {
    candidate.strategy = L1Strategy::NONE;
    candidate.rejectionReason = "Single-pass load, no reuse";
    return candidate;
  }

  // Check 2: Static tile shape
  if (candidate.region.shape.empty()) {
    candidate.strategy = L1Strategy::NONE;
    candidate.rejectionReason = "Dynamic tile shape";
    return candidate;
  }

  // Check 3: NZ compatibility
  if (!candidate.region.isNZCompatible()) {
    candidate.strategy = L1Strategy::NONE;
    candidate.rejectionReason =
        "Tile not NZ-compatible (not divisible by 16x16)";
    return candidate;
  }

  // Check 4: Trip count must be static
  if (candidate.numTiles < 1) {
    candidate.strategy = L1Strategy::NONE;
    candidate.rejectionReason = "Dynamic loop trip count";
    return candidate;
  }

  // Check 5: L1 capacity
  candidate.l1FootprintBytes = candidate.numTiles * candidate.region.byteSize;
  if (candidate.l1FootprintBytes > kL1CapacityBytes) {
    candidate.strategy = L1Strategy::NONE;
    candidate.rejectionReason = "L1 capacity exceeded (" +
                                std::to_string(candidate.l1FootprintBytes) +
                                " > " + std::to_string(kL1CapacityBytes) + ")";
    return candidate;
  }

  // Check 6: Read-only between loops
  // For fused-add patterns, the intermediate (S) is written then re-read.
  // We still allow this but flag it.
  bool hasIntermediateWrite = false;
  for (size_t i = 0; i + 1 < candidate.region.containingLoops.size(); ++i) {
    if (hasStoresBetweenLoops(funcOp, candidate.region.base,
                              candidate.region.containingLoops[i],
                              candidate.region.containingLoops[i + 1])) {
      hasIntermediateWrite = true;
    }
  }
  candidate.region.isReadOnly = !hasIntermediateWrite;

  // NZ dimensions
  int64_t tileElems = 1;
  for (int64_t d : candidate.region.shape)
    tileElems *= d;
  candidate.nzOuterCols = 1; // always 1 for vector kernels
  int64_t outerRowsPerTile = tileElems / kNZBlockDim;
  candidate.nzTotalOuterRows = candidate.numTiles * outerRowsPerTile;

  // GM bytes saved
  candidate.gmBytesRemoved =
      (candidate.reuseCount - 1) * candidate.region.byteSize *
      candidate.numTiles;

  // Strategy and confidence scoring
  candidate.strategy = L1Strategy::CACHE_REUSE;

  if (candidate.region.isReadOnly && candidate.reuseCount >= 2) {
    candidate.confidenceScore = 1.0f; // Matches proven RMSNorm pattern
  } else if (hasIntermediateWrite && candidate.reuseCount >= 2) {
    // Fused-add pattern: S is written then re-read. Still beneficial
    // but the write must happen before L1 staging (or stage after write).
    candidate.confidenceScore = 0.7f;
  } else {
    candidate.confidenceScore = 0.5f;
  }

  // Bonus for higher reuse (3-pass LayerNorm gets higher score)
  if (candidate.reuseCount >= 3) {
    candidate.confidenceScore =
        std::min(1.0f, candidate.confidenceScore + 0.1f);
  }

  return candidate;
}

} // anonymous namespace

// --- Public API ---

/// Collect direct-child scf::ForOp loops of a parent ForOp.
/// "Direct child" means the inner loop's nearest parent ForOp is exactly
/// parentLoop (no intermediate ForOp nesting).
static SmallVector<scf::ForOp>
collectDirectChildLoops(scf::ForOp parentLoop) {
  SmallVector<scf::ForOp> children;
  parentLoop.walk([&](scf::ForOp innerLoop) {
    if (innerLoop.getOperation() == parentLoop.getOperation())
      return;
    if (innerLoop->getParentOfType<scf::ForOp>() == parentLoop)
      children.push_back(innerLoop);
  });
  return children;
}

/// Analyze a set of sibling loops for L1 candidates.
/// Works for both top-level siblings and nested siblings within a parent.
static void analyzeSiblingLoops(func::FuncOp funcOp,
                                ArrayRef<scf::ForOp> siblingLoops,
                                scf::ForOp parentLoop,
                                L1CandidateSet &result) {
  auto allLoads = collectGMLoads(siblingLoops);

  llvm::errs() << "[L1-Analysis] Found " << allLoads.size()
               << " GM loads across " << siblingLoops.size()
               << " sibling loops"
               << (parentLoop ? " (nested)" : " (top-level)") << "\n";

  auto groups = groupLoadsByGMArg(allLoads);

  for (auto &[argIndex, group] : groups) {
    int distinctLoops = countDistinctLoops(group);
    if (distinctLoops >= 2) {
      result.multiPassGroups++;
      L1Candidate candidate =
          buildCandidate(funcOp, argIndex, group, siblingLoops);

      // Nested siblings get slightly lower confidence than top-level
      // (the transform must handle the parent loop context)
      if (parentLoop && candidate.confidenceScore > 0.0f)
        candidate.confidenceScore =
            std::max(0.5f, candidate.confidenceScore - 0.1f);

      llvm::errs() << "[L1-Analysis] GM arg " << argIndex << ": "
                   << distinctLoops << " loops, reuse="
                   << candidate.reuseCount
                   << ", bytes=" << candidate.region.byteSize
                   << ", strategy="
                   << (candidate.strategy == L1Strategy::CACHE_REUSE
                           ? "CACHE_REUSE"
                           : "NONE");
      if (!candidate.rejectionReason.empty())
        llvm::errs() << ", rejected: " << candidate.rejectionReason;
      llvm::errs() << "\n";

      if (candidate.isProfitable())
        result.profitableCandidates++;

      result.candidates.push_back(std::move(candidate));
    }
  }
}

L1CandidateSet analyzeL1Candidates(func::FuncOp funcOp) {
  L1CandidateSet result;
  result.function = funcOp;

  // Step 1: Collect top-level loops (no parent ForOp)
  SmallVector<scf::ForOp> topLevelLoops;
  funcOp.walk([&](scf::ForOp forOp) {
    if (!forOp->getParentOfType<scf::ForOp>())
      topLevelLoops.push_back(forOp);
  });
  result.totalLoops = topLevelLoops.size();

  llvm::errs() << "[L1-Analysis] Function '" << funcOp.getName()
               << "': " << topLevelLoops.size() << " top-level loops\n";

  // Path A: >=2 top-level sibling loops (POC pattern)
  if (topLevelLoops.size() >= 2) {
    llvm::errs() << "[L1-Analysis] Path A: top-level sibling analysis\n";
    analyzeSiblingLoops(funcOp, topLevelLoops, /*parentLoop=*/{}, result);
  }

  // Path B: 1 top-level loop → look for nested sibling loops inside it
  // This handles the common row-tiling pattern where col loops are nested.
  if (topLevelLoops.size() == 1) {
    scf::ForOp outerLoop = topLevelLoops[0];
    auto nestedSiblings = collectDirectChildLoops(outerLoop);

    llvm::errs() << "[L1-Analysis] Path B: " << nestedSiblings.size()
                 << " direct-child loops inside outer loop\n";

    if (nestedSiblings.size() >= 2) {
      analyzeSiblingLoops(funcOp, nestedSiblings, outerLoop, result);
    }
  }

  // Path C: Single loop with GM loads → L1 prefetch opportunity.
  // Even a single load per iteration benefits from L1 staging because
  // ND2NZ (MTE2) overlaps with the previous iteration's store (MTE3).
  if (topLevelLoops.size() >= 1 && result.profitableCandidates == 0) {
    for (auto &loop : topLevelLoops) {
      SmallVector<LoopGMLoad> loads;
      // Collect GM loads from this single loop
      SmallVector<scf::ForOp> singleVec = {loop};
      loads = collectGMLoads(singleVec);

      if (loads.empty())
        continue;

      llvm::errs() << "[L1-Analysis] Path C: single-loop prefetch, "
                   << loads.size() << " GM loads\n";

      for (auto &load : loads) {
        if (!load.ubAlloc)
          continue;

        L1Candidate candidate;
        candidate.region.base = load.gmRoot;
        candidate.region.baseArgIndex = load.gmArgIndex;
        candidate.region.shape = getTileShape(load.ubAlloc);
        candidate.region.elementType =
            load.ubAlloc.getType().getElementType();
        candidate.region.computeByteSize();
        candidate.region.containingLoops.push_back(load.loop);
        candidate.region.loadOps.push_back(load.copyOp);
        candidate.region.isReadOnly = true;
        candidate.reuseCount = 1;

        // Trip count can be dynamic for PREFETCH — we only need 1 tile in L1
        candidate.numTiles = getConstantTripCount(loop);

        if (candidate.region.shape.empty()) {
          candidate.strategy = L1Strategy::NONE;
          candidate.rejectionReason = "Dynamic tile shape";
        } else if (!candidate.region.isNZCompatible()) {
          candidate.strategy = L1Strategy::NONE;
          candidate.rejectionReason =
              "Tile not NZ-compatible (not divisible by 16x16)";
        } else if (candidate.region.byteSize > kL1CapacityBytes) {
          candidate.strategy = L1Strategy::NONE;
          candidate.rejectionReason = "Single tile exceeds L1 capacity";
        } else {
          candidate.strategy = L1Strategy::PREFETCH;
          candidate.l1FootprintBytes = candidate.region.byteSize;
          candidate.gmBytesRemoved = 0; // no cross-loop reuse savings
          candidate.confidenceScore = 0.8f;

          // NZ dimensions for a single tile
          int64_t tileElems = 1;
          for (int64_t d : candidate.region.shape)
            tileElems *= d;
          candidate.nzOuterCols = 1;
          candidate.nzTotalOuterRows = tileElems / kNZBlockDim;
        }

        llvm::errs() << "[L1-Analysis] PREFETCH candidate: arg "
                     << load.gmArgIndex
                     << ", bytes=" << candidate.region.byteSize
                     << ", strategy="
                     << (candidate.strategy == L1Strategy::PREFETCH
                             ? "PREFETCH"
                             : "NONE");
        if (!candidate.rejectionReason.empty())
          llvm::errs() << ", rejected: " << candidate.rejectionReason;
        llvm::errs() << "\n";

        if (candidate.isProfitable())
          result.profitableCandidates++;

        result.candidates.push_back(std::move(candidate));
      }
    }
  }

  llvm::errs() << "[L1-Analysis] Summary: " << result.multiPassGroups
               << " multi-pass groups, " << result.profitableCandidates
               << " profitable candidates\n";

  return result;
}

} // namespace triton
} // namespace mlir
