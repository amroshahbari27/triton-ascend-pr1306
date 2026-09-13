/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier:
 * MIT */
#pragma once
// Defines the canonical SSA graph shared by classification, scheduling, and
// allocation. Physical storage and ownership decisions annotate this graph only
// after scheduling.
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <limits>
#include <optional>
namespace mlir::triton::cv_split {
enum class EngineType { CUBE, VECTOR };
enum class Resource { Cube, FixPipe, Vector, UBMove };
enum class GraphNodeKind { Operation, Pack, Transfer };
enum class GraphEdgeKind {
  SSA,
  TransferSource,
  TransferDestination,
  ResourceOrder,
  Ownership
};
enum class TransferKind { None, L0CToUB, UBToL1 };
enum class StorageSpace { None, L0C, UB, L1 };
inline constexpr unsigned kUnscheduledLevel =
    std::numeric_limits<unsigned>::max();
inline constexpr llvm::StringLiteral kOriginAttr = "cv_split.origin_id";
struct GraphNode {
  Operation *op = nullptr;
  Value value;
  GraphNodeKind kind = GraphNodeKind::Operation;
  Resource resource = Resource::Vector;
  TransferKind transfer = TransferKind::None;
  int64_t origin = -1;
  unsigned sourceOrder = 0, reverseDepth = 0, level = kUnscheduledLevel;
  llvm::SmallVector<unsigned> predecessors, successors;
};
struct GraphEdge {
  unsigned producer = 0, consumer = 0;
  GraphEdgeKind kind = GraphEdgeKind::SSA;
  Value value;
  unsigned iterationDistance = 0;
};
struct StorageInterval {
  unsigned producer = 0, consumer = 0, begin = 0, end = 0;
  StorageSpace space = StorageSpace::None;
  uint64_t bytes = 0;
  unsigned alignment = 1;
  std::optional<unsigned> slot;
  int64_t family = -1;
};
struct PhysicalSlot {
  StorageSpace space = StorageSpace::None;
  uint64_t bytes = 0;
  unsigned alignment = 1, id = 0;
};
struct AIVPartition {
  unsigned dimension = 0;
  int64_t fullExtent = 0, perAIVExtent = 0;
  unsigned parts = 2;
};
struct SSAGraph {
  scf::ForOp loop;
  llvm::SmallVector<GraphNode> nodes;
  llvm::SmallVector<GraphEdge> edges;
  llvm::DenseMap<Operation *, unsigned> operationNodes;
  llvm::SmallVector<StorageInterval> storageIntervals;
  llvm::SmallVector<PhysicalSlot> physicalSlots;
  std::optional<AIVPartition> aivPartition;
};
void addGraphEdge(SSAGraph &, unsigned producer, unsigned consumer,
                  GraphEdgeKind, Value value = {}, unsigned distance = 0);
void setOpEngineTypeAttr(Operation *, EngineType);
void removeEngineTypeAttrs(ModuleOp);
FailureOr<scf::ForOp> prepareSSA(func::FuncOp, int unrollFactor);
LogicalResult applyVFRewriteStage(scf::ForOp, bool enablePatterns);
struct VFTransferSite {
  Value source;
  Operation *ready;
  Value packedDestination;
  Value maximumDestination;
  Value scaledDestination;
  Value sumDestination;
  Value consumedInput;
  Operation *consumptionComplete = nullptr;
};
LogicalResult
    materializeOptionalVFRewritesAfterRowSplit(MutableArrayRef<VFTransferSite>);
FailureOr<SSAGraph> buildSSAGraph(scf::ForOp);
LogicalResult scheduleSSAGraph(SSAGraph &);
LogicalResult lowerSSAToBufferAllocation(func::FuncOp, SSAGraph &);
} // namespace mlir::triton::cv_split
