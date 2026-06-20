/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the MIT license.
 *
 * L1 Cache Analysis — General-purpose analysis pass that detects
 * L1 caching opportunities across all pure vector kernels.
 *
 * Produces L1CandidateSet: a list of GM regions that are loaded
 * in multiple sequential loops and can be staged through L1.
 */

#ifndef TRITON_ADAPTER_L1_CACHE_ANALYSIS_H
#define TRITON_ADAPTER_L1_CACHE_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

#include <string>

namespace mlir {
namespace triton {

enum class L1Strategy {
  CACHE_REUSE,   // Multi-pass: stage GM→L1 on first load, reuse on subsequent
  UB_RELIEF,     // Move read-only data to L1 to free UB capacity
  PREFETCH,      // Prefetch GM→L1 N iterations ahead
  STENCIL_CACHE, // Cache overlapping stencil regions
  NONE           // No L1 opportunity
};

/// Describes a contiguous region of GM being accessed.
struct MemoryRegion {
  Value base;           // GM argument SSA value (func arg or alloc root)
  int64_t baseArgIndex; // Function argument index (-1 if not a func arg)

  /// Static tile shape from BLOCK_SIZE parameters.
  SmallVector<int64_t> shape;

  /// Strides from memref type or subview metadata.
  SmallVector<int64_t> strides;

  /// Element type (f16, f32, i32, etc.).
  Type elementType;

  /// Total bytes for one tile: product(shape) * sizeof(elementType).
  int64_t byteSize = 0;

  /// Whether this region is only read (no stores to same base between uses).
  bool isReadOnly = true;

  /// Whether the offset expression relative to the outer loop IV is the
  /// same across all loops that access this region. Set by analysis.
  bool hasIdenticalOffsets = false;

  /// The scf::ForOp loops that contain loads from this region.
  SmallVector<scf::ForOp> containingLoops;

  /// The actual memref::CopyOp (or load ops) that access this region.
  SmallVector<Operation *> loadOps;

  /// NZ compatibility: tile size must be divisible by 16×16 = 256 elements.
  bool isNZCompatible() const;

  /// Compute byte size from shape and element type.
  void computeByteSize();

  /// Check if two regions access the same GM data.
  static bool isSameRegion(const MemoryRegion &r1, const MemoryRegion &r2);
};

/// A scored L1 caching candidate.
struct L1Candidate {
  MemoryRegion region;

  /// How many times this region is loaded across loops.
  int reuseCount = 0;

  /// Total GM bytes that would be eliminated.
  int64_t gmBytesRemoved = 0;

  /// L1 footprint: byteSize × numTiles (for full-row caching).
  int64_t l1FootprintBytes = 0;

  /// Number of tiles (trip count of the inner col loop).
  int64_t numTiles = 0;

  /// NZ format dimensions for the full L1 allocation.
  int64_t nzOuterCols = 0;
  int64_t nzTotalOuterRows = 0;

  /// Which strategy to apply.
  L1Strategy strategy = L1Strategy::NONE;

  /// Confidence: 1.0 = proven (matches POC pattern exactly),
  ///             0.5-0.9 = high confidence generalization,
  ///             <0.5 = speculative.
  float confidenceScore = 0.0f;

  /// If not profitable or not applicable, why.
  std::string rejectionReason;

  /// Whether this candidate passed all checks and should be transformed.
  bool isProfitable() const {
    return strategy != L1Strategy::NONE && rejectionReason.empty() &&
           confidenceScore > 0.0f;
  }
};

/// Result of L1 cache analysis for a single function.
struct L1CandidateSet {
  SmallVector<L1Candidate, 4> candidates;
  func::FuncOp function;

  /// Summary statistics.
  int totalLoops = 0;
  int multiPassGroups = 0;
  int profitableCandidates = 0;
};

/// Run L1 cache analysis on a function.
/// Returns the set of L1 caching candidates found.
L1CandidateSet analyzeL1Candidates(func::FuncOp funcOp);

/// Apply CACHE_REUSE transform for an L1 candidate.
/// Stages GM→L1 on first load loop, replaces subsequent loops with L1→UB.
bool applyCacheReuseTransform(func::FuncOp funcOp,
                              const L1Candidate &candidate);

/// Apply PREFETCH transform for an L1 candidate.
/// Single-loop pattern: insert ND2NZ (GM→L1) + L12UB (L1→UB) per iteration.
/// Benefit: MTE2 (ND2NZ) overlaps with previous iteration's MTE3 (store).
bool applyPrefetchTransform(func::FuncOp funcOp,
                            const L1Candidate &candidate);

/// Trace a memref value back to its root GM function argument.
/// Returns the BlockArgument if found, or a null Value.
Value traceToGMRoot(Value memrefVal);

/// Get the constant trip count of an scf::ForOp, or -1 if dynamic.
int64_t getConstantTripCount(scf::ForOp loop);

/// L1 capacity limit in bytes (512 KB for A5 NPU).
constexpr int64_t kL1CapacityBytes = 512 * 1024;

/// NZ fractal block dimension.
constexpr int64_t kNZBlockDim = 16;

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_L1_CACHE_ANALYSIS_H
