/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier:
 * MIT */
// Lowers one scheduled SSA graph into typed L0C/UB/L1 buffers, transfers, and
// ownership events. Every storage space uses the scheduled lifetimes and one
// oldest-free allocator.
#include "ascend/include/CVSplitScheduling/Attributes.h"
#include "ascend/include/CVSplitScheduling/BufferSlotPlan.h"
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#define DEBUG_TYPE "cv-split-ssa-buffer-allocation"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[cv-split-buffer] " << X << '\n')
namespace mlir::triton::cv_split {
namespace {
constexpr uint64_t kL0CBytes = 256 * 1024, kUsableUBBytes = 248 * 1024,
                   kL1Bytes = 512 * 1024;
// Physical UB is 256 KiB. Policy reserves 8 KiB, so CV-split may plan 248 KiB.
// A5 exposes block-event ids 0..15.  This pass requires event-free input.
constexpr unsigned kBlockEventCount = 16;
struct SpaceLimit {
  StorageSpace space;
  uint64_t bytes;
  unsigned alignment;
};
constexpr SpaceLimit kLimits[] = {{StorageSpace::L0C, kL0CBytes, 64},
                                  {StorageSpace::UB, kUsableUBBytes, 32},
                                  {StorageSpace::L1, kL1Bytes, 32}};
struct TransferPlan {
  unsigned node, source;
  SmallVector<unsigned> destinations;
  unsigned sourceInterval, destinationInterval;
  int readyFlag = -1;
};
struct ReuseRelation {
  unsigned previousInterval, nextInterval;
  bool loopWrap;
  int flag = -1;
};
struct PhysicalPlan {
  SmallVector<Type> intervalTypes;
  SmallVector<SmallVector<int64_t>> intervalShapes;
  SmallVector<bool> slotRemote;
  SmallVector<TransferPlan> transfers;
  SmallVector<ReuseRelation> reuse;
};
struct TransferEmission {
  Value originalSource, source, ub, l1;
  RankedTensorType fullType;
  unsigned packNode, transferNode;
  int readyFlag;
  Operation *ready = nullptr;
};
struct ScopeResult {
  scope::ScopeOp cubeScope, vectorScope;
  scf::ForOp cubeLoop, vectorLoop;
  DenseMap<Value, Value> cubeValues;
};
EngineType engine(Resource resource) {
  return resource == Resource::Cube || resource == Resource::FixPipe
             ? EngineType::CUBE
             : EngineType::VECTOR;
}
bool isRemote(const SSAGraph &graph, const StorageInterval &interval) {
  return engine(graph.nodes[interval.producer].resource) !=
         engine(graph.nodes[interval.consumer].resource);
}
hivm::PIPE pipe(Resource resource) {
  switch (resource) {
  case Resource::Cube:
    return hivm::PIPE::PIPE_M;
  case Resource::FixPipe:
    return hivm::PIPE::PIPE_FIX;
  case Resource::Vector:
    return hivm::PIPE::PIPE_V;
  case Resource::UBMove:
    return hivm::PIPE::PIPE_MTE3;
  }
  llvm_unreachable("unknown CV-split resource");
}
std::pair<hivm::PIPE, hivm::PIPE> syncPipes(Resource from, Resource to) {
  if (from == Resource::UBMove && to == Resource::Cube)
    return {hivm::PIPE::PIPE_MTE3, hivm::PIPE::PIPE_MTE1};
  if (from == Resource::Cube && to == Resource::UBMove)
    return {hivm::PIPE::PIPE_MTE1, hivm::PIPE::PIPE_MTE3};
  return {pipe(from), pipe(to)};
}
const SpaceLimit *limitFor(StorageSpace space) {
  for (const SpaceLimit &limit : kLimits)
    if (limit.space == space)
      return &limit;
  return nullptr;
}
StringRef storageName(StorageSpace space) {
  switch (space) {
  case StorageSpace::L0C:
    return "L0C";
  case StorageSpace::UB:
    return "UB";
  case StorageSpace::L1:
    return "L1";
  case StorageSpace::None:
    return "None";
  }
  llvm_unreachable("unknown CV-split storage space");
}
std::optional<hivm::AddressSpace> addressSpace(StorageSpace space) {
  switch (space) {
  case StorageSpace::L0C:
    return hivm::AddressSpace::L0C;
  case StorageSpace::UB:
    return hivm::AddressSpace::UB;
  case StorageSpace::L1:
    return hivm::AddressSpace::L1;
  case StorageSpace::None:
    return std::nullopt;
  }
  return std::nullopt;
}
std::optional<StorageSpace> storageSpace(Attribute attribute) {
  auto space = dyn_cast_or_null<hivm::AddressSpaceAttr>(attribute);
  if (!space)
    return std::nullopt;
  switch (space.getAddressSpace()) {
  case hivm::AddressSpace::L0C:
    return StorageSpace::L0C;
  case hivm::AddressSpace::UB:
    return StorageSpace::UB;
  case hivm::AddressSpace::L1:
    return StorageSpace::L1;
  default:
    return std::nullopt;
  }
}
FailureOr<uint64_t> bytesFor(ShapedType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return failure();
  uint64_t elements = 1;
  if (auto memref = dyn_cast<MemRefType>(type)) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memref.getStridesAndOffset(strides, offset)) || offset < 0 ||
        llvm::any_of(strides, [](int64_t stride) { return stride <= 0; }))
      return failure();
    elements = static_cast<uint64_t>(offset) + 1;
    for (auto [extent, stride] : llvm::zip(type.getShape(), strides)) {
      uint64_t tail;
      if (extent <= 0 ||
          !checkedBufferMultiply(static_cast<uint64_t>(extent - 1),
                                 static_cast<uint64_t>(stride), tail) ||
          !checkedBufferAdd(elements, tail, elements))
        return failure();
    }
  } else {
    for (int64_t extent : type.getShape())
      if (extent <= 0 || !checkedBufferMultiply(
                             elements, static_cast<uint64_t>(extent), elements))
        return failure();
  }
  uint64_t bytes, elementBytes = (type.getElementTypeBitWidth() + 7) / 8;
  if (!checkedBufferMultiply(elements, elementBytes, bytes))
    return failure();
  return bytes;
}
bool earlier(const SSAGraph &graph, unsigned lhs, unsigned rhs) {
  const GraphNode &a = graph.nodes[lhs], &b = graph.nodes[rhs];
  return std::tuple(a.level, a.sourceOrder, lhs) <
         std::tuple(b.level, b.sourceOrder, rhs);
}
bool reaches(const SSAGraph &graph, unsigned from, unsigned to) {
  SmallVector<unsigned> pending{from};
  DenseSet<unsigned> seen;
  while (!pending.empty()) {
    unsigned node = pending.pop_back_val();
    if (node == to)
      return true;
    if (!seen.insert(node).second)
      continue;
    llvm::append_range(pending, graph.nodes[node].successors);
  }
  return false;
}
unsigned extreme(const SSAGraph &graph, ArrayRef<unsigned> nodes, bool last) {
  assert(!nodes.empty());
  return *(last ? llvm::max_element(nodes,
                                    [&](unsigned a, unsigned b) {
                                      return earlier(graph, a, b);
                                    })
                : llvm::min_element(nodes, [&](unsigned a, unsigned b) {
                    return earlier(graph, a, b);
                  }));
}
SmallVector<unsigned> edgeEnds(const SSAGraph &graph, unsigned node,
                               GraphEdgeKind kind, bool outgoing) {
  SmallVector<unsigned> result;
  for (const GraphEdge &edge : graph.edges) {
    if (edge.kind == kind && outgoing && edge.producer == node)
      result.push_back(edge.consumer);
    if (edge.kind == kind && !outgoing && edge.consumer == node)
      result.push_back(edge.producer);
  }
  return result;
}
bool aliasesFirstShapedOperand(Operation *op, Value input) {
  if (!op || !op->hasTrait<OpTrait::Elementwise>() ||
      op->getNumResults() != 1 || op->getResult(0).getType() != input.getType())
    return false;
  for (Value operand : op->getOperands()) {
    if (!isa<ShapedType>(operand.getType()))
      continue;
    return operand == input;
  }
  return false;
}
unsigned receivingStorageEnd(const SSAGraph &graph,
                             ArrayRef<unsigned> destinations, Value value) {
  SmallVector<std::pair<unsigned, Value>> pending;
  for (unsigned destination : destinations)
    pending.emplace_back(destination, value);
  DenseSet<std::pair<unsigned, Value>> seen;
  SmallVector<unsigned> readers;
  while (!pending.empty()) {
    auto [node, input] = pending.pop_back_val();
    if (!seen.insert({node, input}).second)
      continue;
    readers.push_back(node);
    Operation *op = graph.nodes[node].op;
    if (!aliasesFirstShapedOperand(op, input))
      continue;
    Value alias = op->getResult(0);
    for (const GraphEdge &edge : graph.edges)
      if (edge.kind == GraphEdgeKind::SSA && edge.producer == node &&
          edge.value == alias &&
          graph.nodes[edge.consumer].resource == graph.nodes[node].resource)
        pending.emplace_back(edge.consumer, alias);
  }
  return extreme(graph, readers, true);
}
LogicalResult addInterval(SSAGraph &graph, PhysicalPlan &plan,
                          unsigned producer, unsigned consumer,
                          StorageSpace space, uint64_t bytes, Type elementType,
                          ArrayRef<int64_t> shape, unsigned &index) {
  const SpaceLimit *limit = limitFor(space);
  if (!limit || !bytes)
    return failure();
  uint64_t aligned;
  if (!checkedBufferAdd(bytes, limit->alignment - 1, aligned))
    return failure();
  aligned = aligned / limit->alignment * limit->alignment;
  index = static_cast<unsigned>(graph.storageIntervals.size());
  graph.storageIntervals.push_back({producer, consumer,
                                    graph.nodes[producer].level,
                                    graph.nodes[consumer].level, space, aligned,
                                    limit->alignment, std::nullopt});
  graph.storageIntervals.back().family = graph.nodes[producer].origin;
  if (graph.storageIntervals.back().end < graph.storageIntervals.back().begin)
    return failure();
  plan.intervalTypes.push_back(elementType);
  plan.intervalShapes.emplace_back(shape);
  return success();
}
FailureOr<PhysicalPlan> buildIntervals(SSAGraph &graph) {
  if (!graph.aivPartition)
    return failure();
  PhysicalPlan plan;
  graph.storageIntervals.clear();
  graph.physicalSlots.clear();
  for (auto [rawNodeIndex, node] : llvm::enumerate(graph.nodes)) {
    unsigned nodeIndex = static_cast<unsigned>(rawNodeIndex);
    if (node.kind != GraphNodeKind::Transfer)
      continue;
    auto type = dyn_cast<RankedTensorType>(node.value.getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
        type.getEncoding() || !type.getElementType().isIntOrFloat())
      return failure();
    SmallVector<unsigned> sources =
        edgeEnds(graph, nodeIndex, GraphEdgeKind::TransferSource, false);
    SmallVector<unsigned> destinations =
        edgeEnds(graph, nodeIndex, GraphEdgeKind::TransferDestination, true);
    if (sources.size() != 1 || destinations.empty())
      return failure();
    unsigned source = sources.front();
    if (graph.nodes[source].origin < 0 || node.origin < 0)
      return failure();
    SmallVector<unsigned> direct{nodeIndex};
    if (graph.nodes[source].kind == GraphNodeKind::Operation)
      for (const GraphEdge &edge : graph.edges)
        if (edge.kind == GraphEdgeKind::SSA && edge.producer == source &&
            edge.value == node.value)
          direct.push_back(edge.consumer);
    unsigned sourceEnd = extreme(graph, direct, true);
    unsigned destinationEnd =
        node.transfer == TransferKind::L0CToUB
            ? receivingStorageEnd(graph, destinations, node.value)
            : extreme(graph, destinations, true);
    const AIVPartition &partition = *graph.aivPartition;
    if (partition.dimension >= type.getRank() ||
        type.getDimSize(partition.dimension) != partition.fullExtent ||
        partition.fullExtent != partition.perAIVExtent * partition.parts)
      return failure();
    SmallVector<int64_t> partShape(type.getShape());
    partShape[partition.dimension] = partition.perAIVExtent;
    auto fullBytes = bytesFor(type);
    auto partBytes =
        bytesFor(RankedTensorType::get(partShape, type.getElementType()));
    if (failed(fullBytes) || failed(partBytes))
      return failure();
    TransferPlan transfer{nodeIndex, source, destinations, 0, 0, -1};
    if (node.transfer == TransferKind::L0CToUB) {
      if (node.resource != Resource::FixPipe ||
          graph.nodes[source].resource != Resource::Cube ||
          llvm::any_of(destinations,
                       [&](unsigned destination) {
                         return graph.nodes[destination].resource !=
                                Resource::Vector;
                       }) ||
          failed(addInterval(graph, plan, source, sourceEnd, StorageSpace::L0C,
                             *fullBytes, type.getElementType(), type.getShape(),
                             transfer.sourceInterval)) ||
          failed(addInterval(graph, plan, nodeIndex, destinationEnd,
                             StorageSpace::UB, *partBytes,
                             type.getElementType(), partShape,
                             transfer.destinationInterval)))
        return failure();
    } else if (node.transfer == TransferKind::UBToL1) {
      if (partition.dimension != 0 || partition.perAIVExtent % kNzTileSize ||
          type.getDimSize(0) % kNzTileSize || type.getDimSize(1) % kNzTileSize)
        return failure();
      // Vector produces the NZ payload as [N/16, M, 16]. Keep that native
      // rank-three object as the planned UB slot and reshape only at UB-to-L1.
      // PlanMemory can then preserve the vector layout and required padding.
      SmallVector<int64_t> packedPart{type.getDimSize(1) / kNzTileSize,
                                      partition.perAIVExtent, kNzTileSize};
      SmallVector<int64_t> packedFull{type.getDimSize(1) / kNzTileSize,
                                      type.getDimSize(0) / kNzTileSize,
                                      kNzTileSize, kNzTileSize};
      if (node.resource != Resource::UBMove ||
          graph.nodes[source].kind != GraphNodeKind::Pack ||
          graph.nodes[source].resource != Resource::Vector ||
          llvm::any_of(destinations,
                       [&](unsigned destination) {
                         return graph.nodes[destination].resource !=
                                Resource::Cube;
                       }) ||
          failed(addInterval(graph, plan, source, nodeIndex, StorageSpace::UB,
                             *partBytes, type.getElementType(), packedPart,
                             transfer.sourceInterval)) ||
          failed(addInterval(graph, plan, nodeIndex, destinationEnd,
                             StorageSpace::L1, *fullBytes,
                             type.getElementType(), packedFull,
                             transfer.destinationInterval)))
        return failure();
    } else {
      return failure();
    }
    plan.transfers.push_back(std::move(transfer));
  }
  if (plan.transfers.empty())
    return failure();
  for (auto [operation, node] : graph.operationNodes) {
    if (graph.nodes[node].resource != Resource::Cube ||
        !isa<linalg::MatmulOp>(operation))
      continue;
    bool tracked = llvm::any_of(plan.transfers, [&](const TransferPlan &item) {
      return item.source == node &&
             graph.nodes[item.node].transfer == TransferKind::L0CToUB;
    });
    if (!tracked) {
      LDBG("untracked matrix storage; preserving fallback");
      return failure();
    }
  }
  llvm::stable_sort(plan.transfers,
                    [&](const TransferPlan &a, const TransferPlan &b) {
                      return earlier(graph, a.node, b.node);
                    });
  return plan;
}
bool aliasesTrackedStorage(Operation *op, Value input) {
  if (aliasesFirstShapedOperand(op, input))
    return true;
  if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(op))
    return view.getViewSource() == input;
  return false;
}
bool collectStorageReaders(SSAGraph &graph, Value value, Resource resource,
                           DenseSet<Value> &seen,
                           SmallVectorImpl<unsigned> &readers) {
  if (!seen.insert(value).second)
    return true;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto node = graph.operationNodes.find(user);
    if (user->getBlock() != graph.loop.getBody() ||
        node == graph.operationNodes.end() ||
        graph.nodes[node->second].resource != resource)
      return false;
    readers.push_back(node->second);
    if (aliasesTrackedStorage(user, value) &&
        !collectStorageReaders(graph, user->getResult(0), resource, seen,
                               readers))
      return false;
  }
  return true;
}
std::optional<std::pair<unsigned, unsigned>>
scheduledAllocationLifetime(memref::AllocOp alloc, SSAGraph &graph) {
  if (alloc->getBlock() != graph.loop.getBody() ||
      !alloc.getDynamicSizes().empty())
    return std::nullopt;
  memref::CopyOp writer;
  SmallVector<bufferization::ToTensorOp> views;
  for (OpOperand &use : alloc.getMemref().getUses()) {
    if (auto copy = dyn_cast<memref::CopyOp>(use.getOwner());
        copy && copy.getTarget() == alloc.getMemref()) {
      if (writer || copy.getSource() == alloc.getMemref())
        return std::nullopt;
      writer = copy;
    } else if (auto view = dyn_cast<bufferization::ToTensorOp>(use.getOwner());
               view && view.getBuffer() == alloc.getMemref()) {
      views.push_back(view);
    } else {
      return std::nullopt;
    }
  }
  if (!writer || views.empty())
    return std::nullopt;
  auto writerNode = graph.operationNodes.find(writer);
  if (writerNode == graph.operationNodes.end())
    return std::nullopt;
  Resource resource = graph.nodes[writerNode->second].resource;
  DenseSet<Value> seen;
  SmallVector<unsigned> readers;
  for (bufferization::ToTensorOp view : views) {
    auto viewNode = graph.operationNodes.find(view);
    if (viewNode == graph.operationNodes.end() ||
        graph.nodes[viewNode->second].resource != resource ||
        !reaches(graph, writerNode->second, viewNode->second))
      return std::nullopt;
    readers.push_back(viewNode->second);
    if (!collectStorageReaders(graph, view.getResult(), resource, seen,
                               readers))
      return std::nullopt;
  }
  unsigned end = extreme(graph, readers, true);
  if (!reaches(graph, writerNode->second, end))
    return std::nullopt;
  return std::pair(graph.nodes[writerNode->second].level,
                   graph.nodes[end].level);
}
struct ExistingAllocation {
  StorageSpace space;
  uint64_t bytes;
  unsigned begin, end;
};
FailureOr<DenseMap<StorageSpace, uint64_t>>
existingPeakBytes(func::FuncOp function, SSAGraph &graph) {
  DenseMap<StorageSpace, uint64_t> permanent;
  SmallVector<ExistingAllocation> scheduled;
  WalkResult result = function.walk([&](memref::AllocOp alloc) {
    auto space = storageSpace(alloc.getType().getMemorySpace());
    // Plain local allocations lower to UB. Explicit L0C/L1/UB spaces retain
    // their declared space; other explicit address spaces are not ours.
    if (!space && !alloc.getType().getMemorySpace())
      space = StorageSpace::UB;
    if (!space)
      return WalkResult::advance();
    auto bytes = bytesFor(alloc.getType());
    const SpaceLimit *limit = limitFor(*space);
    if (failed(bytes) || !limit || !alloc.getDynamicSizes().empty())
      return WalkResult::interrupt();
    uint64_t aligned;
    if (!checkedBufferAdd(*bytes, limit->alignment - 1, aligned))
      return WalkResult::interrupt();
    aligned = aligned / limit->alignment * limit->alignment;
    if (auto lifetime = scheduledAllocationLifetime(alloc, graph)) {
      scheduled.push_back({*space, aligned, lifetime->first, lifetime->second});
      return WalkResult::advance();
    }
    uint64_t total;
    if (!checkedBufferAdd(permanent[*space], aligned, total))
      return WalkResult::interrupt();
    permanent[*space] = total;
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  DenseMap<StorageSpace, uint64_t> peak;
  for (const ExistingAllocation &anchor : scheduled) {
    uint64_t live = 0;
    for (const ExistingAllocation &interval : scheduled) {
      if (interval.space != anchor.space || interval.begin > anchor.begin ||
          interval.end < anchor.begin)
        continue;
      if (!checkedBufferAdd(live, interval.bytes, live))
        return failure();
    }
    peak[anchor.space] = std::max(peak[anchor.space], live);
  }
  DenseMap<StorageSpace, uint64_t> totals = permanent;
  for (const SpaceLimit &limit : kLimits) {
    uint64_t total;
    if (!checkedBufferAdd(totals[limit.space], peak[limit.space], total))
      return failure();
    totals[limit.space] = total;
    LDBG("existing " << storageName(limit.space)
                     << " bytes: permanent=" << permanent[limit.space]
                     << " scheduled-peak=" << peak[limit.space]);
  }
  return totals;
}
LogicalResult allocateIntervals(func::FuncOp function, SSAGraph &graph,
                                PhysicalPlan &plan) {
  if (plan.intervalTypes.size() != graph.storageIntervals.size() ||
      plan.intervalShapes.size() != graph.storageIntervals.size())
    return failure();
  auto existing = existingPeakBytes(function, graph);
  if (failed(existing))
    return failure();
  DenseMap<StorageSpace, uint64_t> used = *existing;
  SmallVector<unsigned> order(graph.storageIntervals.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::stable_sort(order, [&](unsigned a, unsigned b) {
    const StorageInterval &x = graph.storageIntervals[a],
                          &y = graph.storageIntervals[b];
    return std::tuple(x.begin, x.end, a) < std::tuple(y.begin, y.end, b);
  });
  DenseMap<unsigned, unsigned> lastInterval;
  auto sameClass = [&](unsigned a, unsigned b) {
    const StorageInterval &x = graph.storageIntervals[a],
                          &y = graph.storageIntervals[b];
    return x.space == y.space && x.bytes == y.bytes &&
           x.alignment == y.alignment &&
           plan.intervalTypes[a] == plan.intervalTypes[b] &&
           plan.intervalShapes[a] == plan.intervalShapes[b] &&
           x.family == y.family &&
           isRemote(graph, x) == isRemote(graph, y);
  };
  auto createSlot = [&](unsigned index, bool reserved) -> LogicalResult {
    StorageInterval &interval = graph.storageIntervals[index];
    const SpaceLimit *limit = limitFor(interval.space);
    if (!limit)
      return failure();
    uint64_t newTotal;
    bool memoryAllows =
        checkedBufferAdd(used[interval.space], interval.bytes, newTotal) &&
        newTotal <= limit->bytes;
    if (!memoryAllows) {
      LDBG("interval " << index
                       << " exhausted memory capacity while reserving "
                          "compatibility classes");
      return failure();
    }
    bool remote = isRemote(graph, interval);
    unsigned id = static_cast<unsigned>(graph.physicalSlots.size());
    graph.physicalSlots.push_back(
        {interval.space, interval.bytes, interval.alignment, id});
    plan.slotRemote.push_back(remote);
    interval.slot = id;
    used[interval.space] = newTotal;
    lastInterval[id] = index;
    LDBG("interval " << index
                     << (reserved ? " reserved compatibility-class slot "
                                  : " created slot ")
                     << id << (remote ? " with block event" : ""));
    return success();
  };
  SmallVector<SmallVector<unsigned>> classes;
  SmallVector<unsigned> intervalClass(graph.storageIntervals.size());
  for (unsigned index : order) {
    auto found = llvm::find_if(classes, [&](ArrayRef<unsigned> members) {
      return sameClass(index, members.front());
    });
    unsigned id;
    if (found == classes.end()) {
      id = static_cast<unsigned>(classes.size());
      classes.push_back({index});
    } else {
      id = static_cast<unsigned>(found - classes.begin());
      found->push_back(index);
    }
    intervalClass[index] = id;
  }
  SmallVector<unsigned> minimumSlots(classes.size(), 1);
  for (auto [classId, members] : llvm::enumerate(classes)) {
    for (unsigned i = 1; i != members.size(); ++i) {
      const StorageInterval &previous = graph.storageIntervals[members[i - 1]];
      const StorageInterval &next = graph.storageIntervals[members[i]];
      if (previous.end >= next.begin ||
          !reaches(graph, previous.consumer, next.producer)) {
        minimumSlots[classId] = 2;
        break;
      }
    }
  }
  DenseMap<StorageSpace, uint64_t> required;
  for (auto [classId, members] : llvm::enumerate(classes)) {
    const StorageInterval &sample = graph.storageIntervals[members.front()];
    uint64_t bytes, total;
    if (!checkedBufferMultiply(sample.bytes,
                               static_cast<uint64_t>(minimumSlots[classId]),
                               bytes) ||
        !checkedBufferAdd(required[sample.space], bytes, total))
      return failure();
    required[sample.space] = total;
  }
  for (const SpaceLimit &limit : kLimits) {
    uint64_t total;
    if (!checkedBufferAdd(used[limit.space], required[limit.space], total) ||
        total > limit.bytes)
      return failure();
  }
  SmallVector<unsigned> slotsPerClass(classes.size());
  for (unsigned index : order) {
    StorageInterval &interval = graph.storageIntervals[index];
    std::optional<unsigned> selected;
    unsigned oldestEnd = 0;
    unsigned classId = intervalClass[index];
    bool reserved = slotsPerClass[classId] < minimumSlots[classId];
    if (!reserved) {
      for (const PhysicalSlot &slot : graph.physicalSlots) {
        if (!lastInterval.count(slot.id))
          continue;
        unsigned previous = lastInterval.lookup(slot.id);
        if (!sameClass(index, previous))
          continue;
        if (graph.storageIntervals[previous].end >= interval.begin)
          continue;
        if (!selected || graph.storageIntervals[previous].end < oldestEnd ||
            (graph.storageIntervals[previous].end == oldestEnd &&
             slot.id < *selected)) {
          selected = slot.id;
          oldestEnd = graph.storageIntervals[previous].end;
        }
      }
    }
    const SpaceLimit *limit = limitFor(interval.space);
    uint64_t optionalTotal;
    bool canCreate = reserved;
    if (reserved) {
      required[interval.space] -= interval.bytes;
    } else if (limit &&
               checkedBufferAdd(used[interval.space], interval.bytes,
                                optionalTotal) &&
               checkedBufferAdd(optionalTotal, required[interval.space],
                                optionalTotal) &&
               optionalTotal <= limit->bytes) {
      canCreate = true;
    }
    if (canCreate) {
      if (failed(createSlot(index, reserved)))
        return failure();
      ++slotsPerClass[classId];
      continue;
    }
    if (selected) {
      unsigned previous = lastInterval.lookup(*selected);
      interval.slot = *selected;
      plan.reuse.push_back({previous, index, false, -1});
      lastInterval[*selected] = index;
      LDBG("interval " << index << " reused oldest free slot " << *selected);
      continue;
    }
    return failure();
  }
  DenseMap<unsigned, SmallVector<unsigned>> intervalsBySlot;
  for (auto [rawIndex, interval] : llvm::enumerate(graph.storageIntervals)) {
    if (!interval.slot)
      return failure();
    intervalsBySlot[*interval.slot].push_back(static_cast<unsigned>(rawIndex));
  }
  for (auto &[slot, intervals] : intervalsBySlot) {
    llvm::stable_sort(intervals, [&](unsigned a, unsigned b) {
      return std::tuple(graph.storageIntervals[a].begin,
                        graph.storageIntervals[a].end,
                        a) < std::tuple(graph.storageIntervals[b].begin,
                                        graph.storageIntervals[b].end, b);
    });
    // A reverse credit is needed only when another logical value overwrites
    // this remote destination.
    if (plan.slotRemote[slot] && intervals.size() > 1)
      plan.reuse.push_back({intervals.back(), intervals.front(), true, -1});
  }
  // This covers pass-owned and explicit address-spaced allocs; downstream
  // PlanMemory remains authoritative.
  LDBG("owned capacity: L0C=" << used[StorageSpace::L0C] << "/" << kL0CBytes
                              << " UB=" << used[StorageSpace::UB] << "/"
                              << kUsableUBBytes << " L1="
                              << used[StorageSpace::L1] << "/" << kL1Bytes
                              << "; downstream PlanMemory required");
  return success();
}
LogicalResult planOwnership(SSAGraph &graph, PhysicalPlan &plan) {
  for (ReuseRelation &relation : plan.reuse) {
    const StorageInterval &previous =
                              graph.storageIntervals[relation.previousInterval],
                          &next = graph.storageIntervals[relation.nextInterval];
    Resource from = graph.nodes[previous.consumer].resource,
             to = graph.nodes[next.producer].resource;
    bool supported = from == to ||
                     (from == Resource::Vector && to == Resource::FixPipe) ||
                     (from == Resource::Cube && to == Resource::UBMove) ||
                     (from == Resource::FixPipe && to == Resource::Cube) ||
                     (from == Resource::UBMove && to == Resource::Vector);
    if (!supported)
      return failure();
    // Loop-carried ownership is metadata, not a same-iteration DAG edge.
    addGraphEdge(graph, previous.consumer, next.producer,
                 GraphEdgeKind::Ownership, {}, relation.loopWrap ? 1 : 0);
  }
  return success();
}
LogicalResult allocateFlags(func::FuncOp function, const SSAGraph &graph,
                            PhysicalPlan &plan) {
  bool hasEvents = false;
  function.walk([&](Operation *op) {
    hasEvents |=
        isa<hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp,
            hivm::SetFlagOp, hivm::WaitFlagOp>(op);
  });
  if (hasEvents)
    return failure();
  unsigned nextBlock = 0;
  // Readiness belongs to the logical transfer. Physical-slot reuse is ordered
  // separately below by one reverse credit per reused remote slot.
  for (TransferPlan &item : plan.transfers) {
    if (nextBlock == kBlockEventCount)
      return failure();
    const StorageInterval &interval =
        graph.storageIntervals[item.destinationInterval];
    if (!interval.slot || !isRemote(graph, interval) ||
        !reaches(graph, extreme(graph, item.destinations, false),
                 interval.consumer))
      return failure();
    item.readyFlag = static_cast<int>(nextBlock++);
    LDBG("transfer " << item.node << " uses ready event " << item.readyFlag);
  }

  DenseMap<unsigned, int> releaseBySlot;
  for (ReuseRelation &relation : plan.reuse) {
    const StorageInterval &previous =
        graph.storageIntervals[relation.previousInterval];
    const StorageInterval &next = graph.storageIntervals[relation.nextInterval];
    if (previous.slot != next.slot)
      return failure();
    if (!isRemote(graph, previous))
      continue;
    auto release = releaseBySlot.find(*previous.slot);
    if (release == releaseBySlot.end()) {
      if (nextBlock == kBlockEventCount)
        return failure();
      release = releaseBySlot
                    .try_emplace(*previous.slot, static_cast<int>(nextBlock++))
                    .first;
    }
    relation.flag = release->second;
    if (relation.loopWrap) {
      bool protectedWrap =
          llvm::any_of(graph.edges, [&](const GraphEdge &edge) {
            return edge.kind == GraphEdgeKind::Ownership &&
                   edge.producer == previous.consumer &&
                   edge.consumer == next.producer &&
                   edge.iterationDistance == 1;
          });
      if (!protectedWrap)
        return failure();
    } else if (!reaches(graph, previous.consumer, next.producer)) {
      return failure();
    }
  }
  LDBG("assigned " << nextBlock << " block events");
  return success();
}
class SlotPool {
public:
  SlotPool(func::FuncOp function, scf::ForOp loop, SSAGraph &graph,
           const PhysicalPlan &plan) {
    DenseMap<unsigned, SmallVector<unsigned>> bySlot;
    for (auto [index, interval] : llvm::enumerate(graph.storageIntervals)) {
      if (!interval.slot) {
        valid = false;
        return;
      }
      bySlot[*interval.slot].push_back(static_cast<unsigned>(index));
    }
    for (const PhysicalSlot &slot : graph.physicalSlots) {
      SmallVector<unsigned> &intervals = bySlot[slot.id];
      if (intervals.empty()) {
        valid = false;
        return;
      }
      unsigned canonical = intervals.front();
      OpBuilder builder(loop);
      std::optional<EngineType> owner;
      if (!plan.slotRemote[slot.id]) {
        std::optional<unsigned> first;
        bool after = false;
        for (unsigned index : intervals) {
          GraphNode &producer =
              graph.nodes[graph.storageIntervals[index].producer];
          unsigned anchor = graph.storageIntervals[index].producer;
          bool candidateAfter = false;
          if (producer.kind == GraphNodeKind::Pack) {
            SmallVector<unsigned> sources =
                edgeEnds(graph, anchor, GraphEdgeKind::SSA, false);
            if (sources.size() != 1) {
              valid = false;
              return;
            }
            anchor = sources.front();
            candidateAfter = true;
          }
          EngineType candidateOwner = engine(producer.resource);
          if (owner && *owner != candidateOwner) {
            valid = false;
            return;
          }
          owner = candidateOwner;
          if (!first || earlier(graph, anchor, *first)) {
            first = anchor;
            after = candidateAfter;
          } else if (anchor == *first)
            after &= candidateAfter;
        }
        Operation *anchor = first ? graph.nodes[*first].op : nullptr;
        if (!anchor || anchor->getBlock() != loop.getBody()) {
          valid = false;
          return;
        }
        if (after)
          builder.setInsertionPointAfter(anchor);
        else
          builder.setInsertionPoint(anchor);
      }
      const StorageInterval &storage = graph.storageIntervals[canonical];
      auto space = hivm::AddressSpaceAttr::get(function.getContext(),
                                               *addressSpace(storage.space));
      MemRefType type =
          MemRefType::get(plan.intervalShapes[canonical],
                          plan.intervalTypes[canonical], nullptr, space);
      auto allocation = builder.create<memref::AllocOp>(loop.getLoc(), type);
      allocation->setAttr("alignment",
                          builder.getI64IntegerAttr(slot.alignment));
      if (owner)
        setOpEngineTypeAttr(allocation, *owner);
      if (storage.space != StorageSpace::L0C) {
        auto mark = builder.create<annotation::MarkOp>(loop.getLoc(),
                                                       allocation.getResult());
        mark->setAttr("effects", builder.getStrArrayAttr({"write", "read"}));
        if (owner)
          setOpEngineTypeAttr(mark, *owner);
      }
      for (unsigned index : intervals) {
        if (plan.intervalTypes[index] != plan.intervalTypes[canonical] ||
            plan.intervalShapes[index] != plan.intervalShapes[canonical]) {
          valid = false;
          return;
        }
        buffers[index] = allocation;
      }
    }
  }
  bool succeeded() const { return valid; }
  Value view(unsigned interval, ArrayRef<int64_t> shape) {
    Value buffer = buffers.lookup(interval);
    return buffer && cast<MemRefType>(buffer.getType()).getShape() == shape
               ? buffer
               : Value{};
  }

private:
  bool valid = true;
  DenseMap<unsigned, Value> buffers;
};
Type partitionType(Type type, const AIVPartition &partition) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (partition.dimension >= tensor.getRank() ||
        tensor.getDimSize(partition.dimension) != partition.fullExtent)
      return type;
    SmallVector<int64_t> shape(tensor.getShape());
    shape[partition.dimension] = partition.perAIVExtent;
    return RankedTensorType::get(shape, tensor.getElementType(),
                                 tensor.getEncoding());
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (partition.dimension >= memref.getRank() ||
        memref.getDimSize(partition.dimension) != partition.fullExtent)
      return type;
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memref.getStridesAndOffset(strides, offset)))
      return type;
    SmallVector<int64_t> shape(memref.getShape());
    shape[partition.dimension] = partition.perAIVExtent;
    return MemRefType::get(shape, memref.getElementType(), memref.getLayout(),
                           memref.getMemorySpace());
  }
  return type;
}
Value createSubBlockId(scope::ScopeOp scope) {
  Block &block = scope.getBodyRegion().front();
  OpBuilder builder(&block, block.begin());
  auto id = builder.create<hivm::GetSubBlockIdxOp>(scope.getLoc(),
                                                   builder.getI64Type());
  return builder
      .create<arith::IndexCastOp>(scope.getLoc(), builder.getIndexType(), id)
      .getResult();
}
void clonePartitionedInitializers(scope::ScopeOp scope,
                                  const AIVPartition &partition) {
  DenseSet<Operation *> inside;
  scope.walk([&](Operation *op) { inside.insert(op); });
  OpBuilder builder(scope);
  DenseMap<Value, Value> mapped;
  scope.walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      Operation *definition = value.getDefiningOp();
      if (!definition || inside.contains(definition) ||
          partitionType(value.getType(), partition) == value.getType())
        continue;
      Value replacement = mapped.lookup(value);
      if (!replacement) {
        auto type = dyn_cast<RankedTensorType>(
            partitionType(value.getType(), partition));
        if (!type)
          continue;
        if (auto fill = dyn_cast<linalg::FillOp>(definition)) {
          auto empty = builder.create<tensor::EmptyOp>(
              scope.getLoc(), type.getShape(), type.getElementType(),
              type.getEncoding());
          replacement =
              builder
                  .create<linalg::FillOp>(scope.getLoc(), fill.getInputs(),
                                          ValueRange{empty})
                  .getResult(0);
        } else if (isa<tensor::EmptyOp>(definition)) {
          replacement = builder
                            .create<tensor::EmptyOp>(
                                scope.getLoc(), type.getShape(),
                                type.getElementType(), type.getEncoding())
                            .getResult();
        } else {
          continue;
        }
        mapped[value] = replacement;
      }
      operand.set(replacement);
    }
  });
}
LogicalResult retileValues(scope::ScopeOp scope,
                           const AIVPartition &partition) {
  SmallVector<arith::ConstantOp> constants;
  scope.walk([&](arith::ConstantOp op) { constants.push_back(op); });
  for (arith::ConstantOp op : constants) {
    Type type = partitionType(op.getType(), partition);
    if (type == op.getType())
      continue;
    auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dense || !dense.isSplat())
      return failure();
    op.setValueAttr(DenseElementsAttr::get(cast<ShapedType>(type),
                                           dense.getSplatValue<Attribute>()));
    op.getResult().setType(type);
  }
  scope.walk([&](Operation *op) {
    if (isa<memref::ReinterpretCastOp, arith::ConstantOp>(op))
      return;
    for (Value result : op->getResults())
      result.setType(partitionType(result.getType(), partition));
    if (auto loop = dyn_cast<scf::ForOp>(op))
      for (unsigned i = 1; i < loop.getBody()->getNumArguments(); ++i)
        loop.getBody()->getArgument(i).setType(
            partitionType(loop.getBody()->getArgument(i).getType(), partition));
  });
  WalkResult result = scope.walk([&](DestinationStyleOpInterface op) {
    for (OpOperand &init : op.getDpsInitsMutable())
      if (isa<TensorType>(init.get().getType()) &&
          init.get().getType() != op.getTiedOpResult(&init).getType())
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  result = scope.walk([&](scf::ForOp loop) {
    for (auto [init, arg, output] : llvm::zip_equal(
             loop.getInitArgs(), loop.getRegionIterArgs(), loop.getResults()))
      if (init.getType() != arg.getType() || init.getType() != output.getType())
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}
LogicalResult retileStores(scope::ScopeOp scope, Value subBlock,
                           const AIVPartition &partition) {
  SmallVector<bufferization::MaterializeInDestinationOp> stores;
  scope.walk([&](bufferization::MaterializeInDestinationOp op) {
    stores.push_back(op);
  });
  for (auto store : stores) {
    auto viewCast = store.getDest().getDefiningOp<memref::ReinterpretCastOp>();
    if (!viewCast || partition.dimension >= viewCast.getType().getRank() ||
        viewCast.getType().getDimSize(partition.dimension) !=
            partition.fullExtent)
      return failure();
    SmallVector<int64_t> staticStrides;
    int64_t staticOffset;
    if (failed(viewCast.getType().getStridesAndOffset(staticStrides,
                                                      staticOffset)))
      return failure();
    int64_t expectedStride = 1;
    for (unsigned i = viewCast.getType().getRank(); i-- > 0;) {
      if (staticStrides[i] != expectedStride || staticStrides[i] <= 0)
        return failure();
      expectedStride *= viewCast.getType().getDimSize(i);
    }
    OpBuilder builder(store);
    SmallVector<OpFoldResult> offsets = viewCast.getMixedOffsets(),
                              sizes = viewCast.getMixedSizes(),
                              strides = viewCast.getMixedStrides();
    unsigned dim = partition.dimension;
    if (offsets.size() != 1 ||
        sizes.size() != static_cast<size_t>(viewCast.getType().getRank()) ||
        strides.size() != static_cast<size_t>(viewCast.getType().getRank()))
      return failure();
    Value old = getValueOrCreateConstantIndexOp(builder, scope.getLoc(),
                                                offsets.front());
    Value width = builder.create<arith::ConstantIndexOp>(
        scope.getLoc(), partition.perAIVExtent);
    Value stride = builder.create<arith::ConstantIndexOp>(scope.getLoc(),
                                                          staticStrides[dim]);
    Value delta = builder.create<arith::MulIOp>(
        scope.getLoc(), subBlock,
        builder.create<arith::MulIOp>(scope.getLoc(), width, stride));
    offsets.front() = getAsOpFoldResult(
        builder.create<arith::AddIOp>(scope.getLoc(), old, delta));
    sizes[dim] = builder.getIndexAttr(partition.perAIVExtent);
    auto type =
        mlir::cast<MemRefType>(partitionType(viewCast.getType(), partition));
    auto replacement = builder.create<memref::ReinterpretCastOp>(
        scope.getLoc(), type, viewCast.getSource(), offsets.front(), sizes,
        strides);
    store.getDestMutable().set(replacement);
    if (viewCast->use_empty())
      viewCast.erase();
  }
  return success();
}
Operation *emitBlockSignal(OpBuilder &builder, Location loc, Resource from,
                           Resource to, int flag, bool set) {
  auto [fromPipe, toPipe] = syncPipes(from, to);
  EngineType owner = set ? engine(from) : engine(to);
  auto core = hivm::TCoreTypeAttr::get(builder.getContext(),
                                       owner == EngineType::CUBE
                                           ? hivm::TCoreType::CUBE
                                           : hivm::TCoreType::VECTOR);
  auto source = hivm::PipeAttr::get(builder.getContext(), fromPipe);
  auto destination = hivm::PipeAttr::get(builder.getContext(), toPipe);
  Operation *operation =
      set ? builder
                .create<hivm::SyncBlockSetOp>(loc, core, source, destination,
                                              builder.getI64IntegerAttr(flag))
                .getOperation()
          : builder
                .create<hivm::SyncBlockWaitOp>(loc, core, source, destination,
                                               builder.getI64IntegerAttr(flag))
                .getOperation();
  setOpEngineTypeAttr(operation, owner);
  return operation;
}
Operation *emitOwnershipSignal(OpBuilder &builder, Location loc, Resource from,
                               Resource to, int flag, bool set) {
  if (engine(from) != engine(to))
    return emitBlockSignal(builder, loc, from, to, flag, set);
  auto [fromPipe, toPipe] = syncPipes(from, to);
  auto source = hivm::PipeAttr::get(builder.getContext(), fromPipe);
  auto destination = hivm::PipeAttr::get(builder.getContext(), toPipe);
  auto event = hivm::EventAttr::get(builder.getContext(),
                                    static_cast<hivm::EVENT>(flag));
  Operation *operation = set ? builder
                                   .create<hivm::SetFlagOp>(
                                       loc, source, destination, event, Value{})
                                   .getOperation()
                             : builder
                                   .create<hivm::WaitFlagOp>(
                                       loc, source, destination, event, Value{})
                                   .getOperation();
  setOpEngineTypeAttr(operation, engine(from));
  return operation;
}
// Removes a tensor binding of a planned buffer that nothing ended up using, so
// a declined optional rewrite leaves no dead view behind.
void discardUnusedBinding(Value binding) {
  Operation *tensor = binding.getDefiningOp();
  if (!tensor || !binding.use_empty())
    return;
  auto cast = tensor->getOperand(0).getDefiningOp<memref::MemorySpaceCastOp>();
  tensor->erase();
  if (cast && cast->use_empty())
    cast->erase();
}
Value plainTensor(OpBuilder &builder, Location loc, Value buffer,
                  RankedTensorType type, EngineType owner) {
  auto memref = cast<MemRefType>(buffer.getType());
  auto plain = MemRefType::get(memref.getShape(), memref.getElementType(),
                               memref.getLayout());
  auto cast = builder.create<memref::MemorySpaceCastOp>(loc, plain, buffer);
  setOpEngineTypeAttr(cast, owner);
  auto tensor =
      builder.create<bufferization::ToTensorOp>(loc, type, cast, true, true);
  setOpEngineTypeAttr(tensor, owner);
  return tensor;
}

Value allocateTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                     StringRef role, EngineType owner) {
  Location storageLoc = NameLoc::get(builder.getStringAttr(role), loc);
  auto space = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB);
  auto memref =
      MemRefType::get(type.getShape(), type.getElementType(), nullptr, space);
  auto allocation = builder.create<memref::AllocOp>(storageLoc, memref);
  allocation->setAttr("alignment", builder.getI64IntegerAttr(32));
  setOpEngineTypeAttr(allocation, owner);
  auto mark =
      builder.create<annotation::MarkOp>(storageLoc, allocation.getResult());
  mark->setAttr("effects", builder.getStrArrayAttr({"write", "read"}));
  setOpEngineTypeAttr(mark, owner);
  return plainTensor(builder, storageLoc, allocation, type, owner);
}
LogicalResult bindL0C(SSAGraph &graph, const PhysicalPlan &plan,
                      SlotPool &pool) {
  DenseMap<Type, Value> zeros;
  for (const TransferPlan &transfer : plan.transfers) {
    if (graph.nodes[transfer.node].transfer != TransferKind::L0CToUB)
      continue;
    auto matmul =
        dyn_cast_or_null<linalg::MatmulOp>(graph.nodes[transfer.source].op);
    auto type =
        dyn_cast<RankedTensorType>(graph.nodes[transfer.node].value.getType());
    auto dps = dyn_cast<DestinationStyleOpInterface>(matmul.getOperation());
    if (!matmul || !type || !dps || dps.getNumDpsInits() != 1)
      return failure();
    Value view = pool.view(transfer.sourceInterval, type.getShape());
    if (!view)
      return failure();
    OpBuilder builder(matmul);
    Value tensor = builder
                       .create<bufferization::ToTensorOp>(
                           graph.loop.getLoc(), type, view, true, true)
                       .getResult();
    setOpEngineTypeAttr(tensor.getDefiningOp(), EngineType::CUBE);
    Value zero = zeros.lookup(type.getElementType());
    if (!zero) {
      zero =
          builder
              .create<arith::ConstantOp>(
                  matmul.getLoc(), builder.getZeroAttr(type.getElementType()))
              .getResult();
      zeros[type.getElementType()] = zero;
      setOpEngineTypeAttr(zero.getDefiningOp(), EngineType::CUBE);
    }
    auto fill = builder.create<linalg::FillOp>(
        matmul.getLoc(), ValueRange{zero}, ValueRange{tensor});
    setOpEngineTypeAttr(fill, EngineType::CUBE);
    Value oldInit = dps.getDpsInitOperand(0)->get();
    dps.getDpsInitOperand(0)->set(fill.getResult(0));
    if (Operation *old = oldInit.getDefiningOp();
        old && isOpTriviallyDead(old)) {
      if (auto node = graph.operationNodes.find(old);
          node != graph.operationNodes.end()) {
        graph.nodes[node->second].op = nullptr;
        graph.operationNodes.erase(node);
      }
      old->erase();
    }
  }
  return success();
}
LogicalResult emitC2V(SSAGraph &graph, const TransferPlan &transfer,
                      SlotPool &pool,
                      DenseMap<unsigned, Operation *> &vectorAnchors,
                      DenseMap<Value, unsigned> &readIntervals) {
  Location loc = graph.loop.getLoc();
  Value value = graph.nodes[transfer.node].value;
  auto type = cast<RankedTensorType>(value.getType());
  auto partType =
      cast<RankedTensorType>(partitionType(type, *graph.aivPartition));
  Operation *producer = graph.nodes[transfer.source].op;
  Value ub = pool.view(transfer.destinationInterval, partType.getShape());
  if (!ub)
    return failure();
  OpBuilder builder(producer);
  builder.setInsertionPointAfter(producer);
  Value scale;
  hivm::FixpipePreQuantModeAttr quant;
  auto fixpipe = builder.create<hivm::FixpipeOp>(
      loc, TypeRange{}, value, ub, ValueRange{},
      hivm::FixpipeDMAModeAttr::get(builder.getContext(),
                                    hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipeDualDstModeAttr::get(builder.getContext(),
                                        hivm::FixpipeDualDstMode::ROW_SPLIT),
      nullptr, quant, nullptr, nullptr, nullptr, scale, ArrayAttr{}, nullptr);
  setOpEngineTypeAttr(fixpipe, EngineType::CUBE);
  graph.nodes[transfer.node].op = fixpipe;
  builder.setInsertionPointAfter(fixpipe);
  emitBlockSignal(builder, loc, Resource::FixPipe, Resource::Vector,
                  transfer.readyFlag, true);
  unsigned firstNode = extreme(graph, transfer.destinations, false);
  Operation *first = graph.nodes[firstNode].op;
  builder.setInsertionPoint(first);
  Operation *wait =
      emitBlockSignal(builder, loc, Resource::FixPipe, Resource::Vector,
                      transfer.readyFlag, false);
  vectorAnchors.try_emplace(firstNode, wait);
  Value tensor = plainTensor(builder, loc, ub, partType, EngineType::VECTOR);
  readIntervals.try_emplace(tensor, transfer.destinationInterval);
  for (unsigned destination : transfer.destinations)
    graph.nodes[destination].op->replaceUsesOfWith(value, tensor);
  return success();
}
FailureOr<TransferEmission>
emitV2C(SSAGraph &graph, const TransferPlan &transfer, SlotPool &pool) {
  Location loc = graph.loop.getLoc();
  Value value = graph.nodes[transfer.node].value;
  auto type = cast<RankedTensorType>(value.getType());
  const AIVPartition &partition = *graph.aivPartition;
  if (partition.dimension != 0 || partition.perAIVExtent % kNzTileSize ||
      type.getDimSize(0) % kNzTileSize || type.getDimSize(1) % kNzTileSize)
    return failure();
  int64_t m = type.getDimSize(0);
  int64_t n16 = type.getDimSize(1) / kNzTileSize;
  int64_t m16 = m / kNzTileSize;
  Value ub = pool.view(transfer.sourceInterval,
                       {n16, partition.perAIVExtent, kNzTileSize});
  Value l1 = pool.view(transfer.destinationInterval,
                       {n16, m16, kNzTileSize, kNzTileSize});
  if (!ub || !l1)
    return failure();
  // The Pack node's Vector operations, UBMove, and Cube views are emitted after
  // scope separation.
  return TransferEmission{value,
                          value,
                          ub,
                          l1,
                          type,
                          transfer.source,
                          transfer.node,
                          transfer.readyFlag,
                          nullptr};
}
LogicalResult stripBlock(Block &block, EngineType keep) {
  Operation *terminator = block.getTerminator();
  SmallVector<Operation *> erase;
  for (Operation &op : block) {
    if (&op == terminator || isa<scf::ForOp>(op))
      continue;
    auto scalar = [](Type type) { return type.isIntOrIndexOrFloat(); };
    bool scalarOnly = op.getNumResults() && op.getNumRegions() == 0 &&
                      isMemoryEffectFree(&op) &&
                      llvm::all_of(op.getOperandTypes(), scalar) &&
                      llvm::all_of(op.getResultTypes(), scalar);
    if (isa<arith::ConstantOp>(op) || scalarOnly) {
      setOpEngineTypeAttr(&op, keep);
      continue;
    }
    auto attr = op.getAttrOfType<StringAttr>("ssbuffer.core_type");
    if (!attr || (attr.getValue() != "CUBE" && attr.getValue() != "VECTOR"))
      return failure();
    EngineType owner =
        attr.getValue() == "CUBE" ? EngineType::CUBE : EngineType::VECTOR;
    if (owner != keep)
      erase.push_back(&op);
  }
  DenseSet<Operation *> eraseSet(erase.begin(), erase.end());
  for (Operation *op : erase)
    for (Value result : op->getResults())
      for (OpOperand &use : result.getUses())
        if (!eraseSet.contains(use.getOwner()) &&
            use.getOwner() != terminator) {
          LDBG("cross-scope " << op->getName() << " result used by retained "
                              << use.getOwner()->getName());
          return failure();
        }
  for (Operation *op : llvm::reverse(erase)) {
    for (Value result : op->getResults()) {
      SmallVector<OpOperand *> yields;
      for (OpOperand &use : result.getUses())
        if (use.getOwner() == terminator)
          yields.push_back(&use);
      if (yields.empty())
        continue;
      auto loop = dyn_cast_or_null<scf::ForOp>(block.getParentOp());
      if (!loop)
        return failure();
      for (OpOperand *use : yields) {
        unsigned index = use->getOperandNumber();
        if (index >= loop.getNumRegionIterArgs() ||
            loop.getRegionIterArgs()[index].getType() != result.getType())
          return failure();
        use->set(loop.getRegionIterArgs()[index]);
      }
    }
    if (!op->use_empty())
      return failure();
    op->erase();
  }
  return success();
}
LogicalResult stripScope(scope::ScopeOp scope, EngineType keep) {
  if (failed(stripBlock(scope.getBodyRegion().front(), keep)))
    return failure();
  WalkResult result = scope.walk([&](scf::ForOp loop) {
    return failed(stripBlock(*loop.getBody(), keep)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}
scf::ForOp trimLoop(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<unsigned> keep;
  for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i) {
    Value arg = loop.getRegionIterArgs()[i];
    bool passThrough = loop.getResult(i).use_empty() && arg.hasOneUse() &&
                       yield.getOperand(i) == arg;
    if (!passThrough && (!arg.use_empty() || !loop.getResult(i).use_empty()))
      keep.push_back(i);
  }
  if (keep.size() == loop.getNumRegionIterArgs())
    return loop;
  SmallVector<Value> init;
  for (unsigned index : keep)
    init.push_back(loop.getInitArgs()[index]);
  OpBuilder builder(loop);
  auto replacement =
      builder.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                 loop.getUpperBound(), loop.getStep(), init);
  replacement->setAttrs(loop->getAttrs());
  Block *oldBody = loop.getBody();
  Block *newBody = replacement.getBody();
  loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
  for (auto [newIndex, oldIndex] : llvm::enumerate(keep))
    loop.getRegionIterArgs()[oldIndex].replaceAllUsesWith(
        replacement.getRegionIterArgs()[newIndex]);
  SmallVector<Value> next;
  for (unsigned index : keep)
    next.push_back(yield.getOperand(index));
  newBody->clear();
  while (&oldBody->front() != yield.getOperation())
    oldBody->front().moveBefore(newBody, newBody->end());
  OpBuilder::atBlockEnd(newBody).create<scf::YieldOp>(yield.getLoc(), next);
  for (auto [newIndex, oldIndex] : llvm::enumerate(keep))
    loop.getResult(oldIndex).replaceAllUsesWith(
        replacement.getResult(newIndex));
  loop.erase();
  return replacement;
}
FailureOr<ScopeResult> separateScopes(func::FuncOp function, SSAGraph &graph) {
  scf::ForOp loop = graph.loop;
  Block *parent = loop->getBlock();
  SmallVector<Operation *> epilogue;
  bool afterLoop = false;
  for (Operation &op : parent->without_terminator()) {
    if (afterLoop)
      epilogue.push_back(&op);
    if (&op == loop.getOperation())
      afterLoop = true;
  }
  for (Operation *op : epilogue) {
    auto attr = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
    if (!attr || attr.getValue() != "VECTOR")
      return failure();
  }
  OpBuilder builder(loop);
  auto cube = builder.create<scope::ScopeOp>(loop.getLoc(), ArrayRef<Type>{});
  cube.getBodyRegion().emplaceBlock();
  cube->setAttr("noinline", UnitAttr::get(function.getContext()));
  Block &cubeBlock = cube.getBodyRegion().front();
  OpBuilder cubeBuilder(&cubeBlock, cubeBlock.end());
  IRMapping mapping;
  auto cubeLoop = cast<scf::ForOp>(cubeBuilder.clone(*loop, mapping));
  DenseMap<Value, Value> cubeValues;
  for (GraphNode &node : graph.nodes)
    if (node.op) {
      Operation *original = node.op;
      for (Value result : original->getResults())
        if (Value clone = mapping.lookupOrNull(result))
          cubeValues[result] = clone;
      if (engine(node.resource) == EngineType::CUBE) {
        Operation *clone = mapping.lookupOrNull(original);
        if (!clone)
          return failure();
        node.op = clone;
      }
    }
  cubeBuilder.create<scope::ReturnOp>(loop.getLoc());
  builder.setInsertionPoint(loop);
  auto vector = builder.create<scope::ScopeOp>(loop.getLoc(), ArrayRef<Type>{});
  vector.getBodyRegion().emplaceBlock();
  vector->setAttr("noinline", UnitAttr::get(function.getContext()));
  Block &vectorBlock = vector.getBodyRegion().front();
  loop->remove();
  vectorBlock.push_back(loop);
  for (Operation *op : epilogue) {
    op->remove();
    vectorBlock.push_back(op);
  }
  OpBuilder(&vectorBlock, vectorBlock.end())
      .create<scope::ReturnOp>(loop.getLoc());
  cube->setAttr(
      hivm::TCoreTypeAttr::name,
      hivm::TCoreTypeAttr::get(function.getContext(), hivm::TCoreType::CUBE));
  vector->setAttr(
      hivm::TCoreTypeAttr::name,
      hivm::TCoreTypeAttr::get(function.getContext(), hivm::TCoreType::VECTOR));
  // Strip Cube after materializeV2C replaces cloned Vector producers in V2C
  // destinations.
  if (failed(stripScope(vector, EngineType::VECTOR)))
    return failure();
  ScopeResult result{cube, vector, cubeLoop, trimLoop(loop),
                     std::move(cubeValues)};
  graph.loop = result.vectorLoop;
  return result;
}
Operation *
scheduledVectorAnchor(SSAGraph &graph, unsigned node,
                      const DenseMap<unsigned, Operation *> &anchors) {
  std::optional<unsigned> best;
  for (auto [rawIndex, candidate] : llvm::enumerate(graph.nodes)) {
    unsigned index = static_cast<unsigned>(rawIndex);
    if (candidate.kind != GraphNodeKind::Operation || !candidate.op ||
        engine(candidate.resource) != EngineType::VECTOR ||
        candidate.op->getBlock() != graph.loop.getBody() ||
        !earlier(graph, node, index))
      continue;
    if (!best || earlier(graph, index, *best))
      best = index;
  }
  return best ? anchors.lookup(*best) ? anchors.lookup(*best)
                                      : graph.nodes[*best].op
              : graph.loop.getBody()->getTerminator();
}

LogicalResult
prepareV2CVFRewrites(SSAGraph &graph,
                     MutableArrayRef<TransferEmission> emissions,
                     const DenseMap<unsigned, Operation *> &anchors,
                     DenseMap<Value, Operation *> &consumerCompletions) {
  if (emissions.empty())
    return success();
  SmallVector<Value> sources;
  for (TransferEmission &emission : emissions)
    sources.push_back(emission.source);
  // Whether a rewrite applies is a pattern question, so it is asked, not
  // answered, here.  Nothing below inspects the computation that produced the
  // published value.
  if (!optionalVFRewriteClaims(sources))
    return success();

  Block *body = graph.loop.getBody();
  SmallVector<VFTransferSite> sites;
  SmallVector<Value> prepared;
  for (TransferEmission &emission : emissions) {
    Operation *source = emission.source.getDefiningOp();
    if (!source || source->getBlock() != body)
      return failure();
    OpBuilder signalBuilder(source);
    signalBuilder.setInsertionPointAfter(source);
    emission.ready = emitBlockSignal(signalBuilder, emission.source.getLoc(),
                                     Resource::UBMove, Resource::Cube,
                                     emission.readyFlag, true);
    auto sourceType = dyn_cast<RankedTensorType>(emission.source.getType());
    auto ubType = dyn_cast<MemRefType>(emission.ub.getType());
    if (!sourceType || sourceType.getRank() != 2 || !ubType ||
        ubType.getRank() != 3 || sourceType.getDimSize(0) <= 0 ||
        sourceType.getDimSize(1) <= 0 || sourceType.getDimSize(1) % kNzTileSize)
      return failure();
    Operation *allocation = emission.ub.getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp>(allocation) ||
        allocation->getBlock() != body)
      return failure();
    // A rewrite may write the destination earlier than the original producer
    // did, so the planned storage is hoisted to the top of the body.  The
    // allocation has no operands, so this is always legal, and it changes no
    // space, size, slot or event decision.
    if (allocation != &body->front())
      allocation->moveBefore(&body->front());
    Operation *last = allocation;
    SmallVector<Operation *> marks;
    for (Operation *user : allocation->getUsers())
      if (isa<annotation::MarkOp>(user))
        marks.push_back(user);
    for (Operation *mark : marks) {
      mark->moveAfter(last);
      last = mark;
    }
    int64_t rows = sourceType.getDimSize(0);
    int64_t n16 = sourceType.getDimSize(1) / kNzTileSize;
    auto packedType = RankedTensorType::get({n16, rows, kNzTileSize},
                                            sourceType.getElementType());
    if (ubType.getShape() != packedType.getShape())
      return failure();
    OpBuilder views(last);
    views.setInsertionPointAfter(last);
    Value destination = plainTensor(views, emission.source.getLoc(), emission.ub,
                                    packedType, EngineType::VECTOR);
    prepared.push_back(destination);
    sites.push_back({emission.source, emission.ready, destination});
  }
  if (failed(materializeOptionalVFRewritesAfterRowSplit(sites)))
    return failure();
  // Whatever the rewrite produced must be the packed destination layout.
  auto packed = [](const VFTransferSite &site) {
    auto type = dyn_cast<RankedTensorType>(site.source.getType());
    return type && type.getRank() == 3;
  };
  // The rewrite is optional even when it is enabled and recognises the sources.
  // A group it declines is published unchanged through the ordinary path; only
  // a partially rewritten group is a failure.
  if (llvm::none_of(sites, packed)) {
    for (Value destination : llvm::reverse(prepared))
      discardUnusedBinding(destination);
    LDBG("optional VF rewrite declined; publishing through the ordinary path");
    return success();
  }
  if (!llvm::all_of(sites, packed))
    return failure();
  for (auto [emission, site] : llvm::zip_equal(emissions, sites)) {
    emission.source = site.source;
    if (site.consumedInput && site.consumptionComplete)
      consumerCompletions.try_emplace(site.consumedInput,
                                      site.consumptionComplete);
  }
  return success();
}

// Physical C2V reads are emitted before late SSA rewrites.  Re-anchor each
// wait/view/tensor group at the first surviving consumer so readiness is
// waited on as late as correctness permits and UB residency starts there.
LogicalResult placeC2VReadsAtFirstUse(scf::ForOp loop) {
  constexpr llvm::StringLiteral kConsumerGroupStart =
      "cv_split.vf_consumer_group_start";
  struct ReadGroup {
    Operation *wait;
    Operation *cast;
    Operation *tensor;
    Operation *anchor;
  };
  Block *body = loop.getBody();
  SmallVector<ReadGroup> groups;
  for (Operation &operation : body->without_terminator()) {
    auto tensor = dyn_cast<bufferization::ToTensorOp>(operation);
    auto cast = tensor
                    ? tensor.getBuffer().getDefiningOp<memref::MemorySpaceCastOp>()
                    : memref::MemorySpaceCastOp();
    Operation *wait = cast ? cast->getPrevNode() : nullptr;
    if (!tensor || !cast || cast->getBlock() != body ||
        tensor->getBlock() != body || !isa_and_nonnull<hivm::SyncBlockWaitOp>(wait) ||
        wait->getNextNode() != cast || cast->getNextNode() != tensor)
      continue;
    Operation *firstUse = nullptr;
    for (OpOperand &use : tensor.getResult().getUses()) {
      Operation *owner = use.getOwner();
      while (owner && owner->getBlock() != body)
        owner = owner->getParentOp();
      if (!owner || owner == tensor || owner->getBlock() != body)
        return failure();
      if (!firstUse || owner->isBeforeInBlock(firstUse))
        firstUse = owner;
    }
    if (!firstUse)
      return failure();
    Operation *anchor = firstUse;
    if (auto add = dyn_cast<arith::AddFOp>(firstUse)) {
      Value other = add.getLhs() == tensor.getResult() ? add.getRhs()
                    : add.getRhs() == tensor.getResult() ? add.getLhs()
                                                          : Value();
      auto multiply = other ? other.getDefiningOp<arith::MulFOp>()
                            : arith::MulFOp();
      if (multiply)
        for (Value operand : multiply->getOperands())
          if (auto broadcast = operand.getDefiningOp<linalg::BroadcastOp>();
              broadcast && broadcast->hasAttr(kConsumerGroupStart) &&
              broadcast->getBlock() == body &&
              broadcast->isBeforeInBlock(firstUse)) {
            anchor = broadcast;
            broadcast->removeAttr(kConsumerGroupStart);
            break;
          }
    }
    groups.push_back({wait, cast, tensor, anchor});
  }
  for (const ReadGroup &group : groups) {
    group.wait->moveBefore(group.anchor);
    group.cast->moveBefore(group.anchor);
    group.tensor->moveBefore(group.anchor);
  }
  LDBG("placed " << groups.size() << " C2V reads at first use");
  return success();
}

LogicalResult materializeV2C(SSAGraph &graph,
                             MutableArrayRef<TransferEmission> emissions,
                             ScopeResult &scopes, const AIVPartition &partition,
                             const DenseMap<unsigned, Operation *> &anchors,
                             Value subBlock) {
  if (partition.dimension != 0 || partition.perAIVExtent % kNzTileSize)
    return failure();
  for (TransferEmission &emission : emissions) {
    if (graph.nodes[emission.packNode].kind != GraphNodeKind::Pack ||
        graph.nodes[emission.transferNode].kind != GraphNodeKind::Transfer ||
        !earlier(graph, emission.packNode, emission.transferNode))
      return failure();
    auto partType = dyn_cast<RankedTensorType>(emission.source.getType());
    RankedTensorType fullType = emission.fullType;
    if (!partType || !partType.hasStaticShape() || fullType.getRank() != 2 ||
        partType.getElementType() != fullType.getElementType() ||
        fullType.getDimSize(0) != partition.fullExtent ||
        fullType.getDimSize(1) <= 0 || fullType.getDimSize(1) % kNzTileSize)
      return failure();
    int64_t m = partition.perAIVExtent;
    int64_t n16 = fullType.getDimSize(1) / kNzTileSize;
    int64_t m16 = m / kNzTileSize;
    Operation *packAnchor =
        scheduledVectorAnchor(graph, emission.packNode, anchors);
    OpBuilder builder(packAnchor);
    auto ubType = cast<MemRefType>(emission.ub.getType());
    Value packedUB;
    Operation *packOperation = nullptr;
    if (partType.getRank() == 2) {
      if (partType.getDimSize(0) != m ||
          partType.getDimSize(1) != fullType.getDimSize(1) ||
          partType.getDimSize(0) % kNzTileSize ||
          partType.getDimSize(1) % kNzTileSize)
        return failure();
      auto shape3Type = RankedTensorType::get({3}, builder.getI64Type());
      auto shape3 = builder.create<arith::ConstantOp>(
          emission.source.getLoc(), shape3Type,
          DenseElementsAttr::get(shape3Type,
                                 ArrayRef<int64_t>{m, n16, kNzTileSize}));
      auto reshaped = builder.create<tensor::ReshapeOp>(
          emission.source.getLoc(),
          RankedTensorType::get({m, n16, kNzTileSize},
                                partType.getElementType()),
          emission.source, shape3);
      auto destinationType = RankedTensorType::get({n16, m, kNzTileSize},
                                                   partType.getElementType());
      Value destination =
          plainTensor(builder, emission.source.getLoc(), emission.ub,
                      destinationType, EngineType::VECTOR);
      auto transpose = builder.create<linalg::TransposeOp>(
          emission.source.getLoc(), reshaped, destination,
          ArrayRef<int64_t>{1, 0, 2});
      auto shape4Type = RankedTensorType::get({4}, builder.getI64Type());
      auto shape4 = builder.create<arith::ConstantOp>(
          emission.source.getLoc(), shape4Type,
          DenseElementsAttr::get(
              shape4Type,
              ArrayRef<int64_t>{n16, m16, kNzTileSize, kNzTileSize}));
      auto packed = builder.create<tensor::ReshapeOp>(
          emission.source.getLoc(),
          RankedTensorType::get({n16, m16, kNzTileSize, kNzTileSize},
                                partType.getElementType()),
          transpose->getResult(0), shape4);
      auto plainPackedType = MemRefType::get(
          packed.getType().getShape(), ubType.getElementType());
      auto packedBuffer = builder.create<bufferization::ToBufferOp>(
          emission.source.getLoc(), plainPackedType, packed);
      auto packedUBType = MemRefType::get(
          packed.getType().getShape(), ubType.getElementType(), nullptr,
          ubType.getMemorySpace());
      packedUB = builder.create<memref::MemorySpaceCastOp>(
          emission.source.getLoc(), packedUBType, packedBuffer);
      for (Operation *op :
           {shape3.getOperation(), reshaped.getOperation(),
             transpose.getOperation(), shape4.getOperation(), packed.getOperation(),
             packedBuffer.getOperation(), packedUB.getDefiningOp()})
        setOpEngineTypeAttr(op, EngineType::VECTOR);
      packOperation = transpose;
    } else if (partType.getRank() == 3) {
      if (partType.getShape() != ArrayRef<int64_t>{n16, m, kNzTileSize})
        return failure();
      Operation *source = emission.source.getDefiningOp();
      if (!source || source->getBlock() != graph.loop.getBody())
        return failure();
      builder.setInsertionPointAfter(source);
      auto shape4Type = RankedTensorType::get({4}, builder.getI64Type());
      auto shape4 = builder.create<arith::ConstantOp>(
          emission.source.getLoc(), shape4Type,
          DenseElementsAttr::get(
              shape4Type,
              ArrayRef<int64_t>{n16, m16, kNzTileSize, kNzTileSize}));
      auto packed = builder.create<tensor::ReshapeOp>(
          emission.source.getLoc(),
          RankedTensorType::get({n16, m16, kNzTileSize, kNzTileSize},
                                partType.getElementType()),
          emission.source, shape4);
      auto plainPackedType = MemRefType::get(
          packed.getType().getShape(), ubType.getElementType());
      auto packedBuffer = builder.create<bufferization::ToBufferOp>(
          emission.source.getLoc(), plainPackedType, packed);
      auto packedUBType = MemRefType::get(
          packed.getType().getShape(), ubType.getElementType(), nullptr,
          ubType.getMemorySpace());
      packedUB = builder.create<memref::MemorySpaceCastOp>(
          emission.source.getLoc(), packedUBType, packedBuffer);
      for (Operation *op :
           {shape4.getOperation(), packed.getOperation(),
            packedBuffer.getOperation(), packedUB.getDefiningOp()})
        setOpEngineTypeAttr(op, EngineType::VECTOR);
      packOperation = packed;
    } else {
      return failure();
    }
    graph.nodes[emission.packNode].op = packOperation;
    Operation *moveAnchor =
        scheduledVectorAnchor(graph, emission.transferNode, anchors);
    // A rewritten rank-three producer is the real scheduled producer.  Emit
    // its asynchronous move immediately, rather than using the stale anchor
    // that preceded VF materialization.
    if (partType.getRank() == 3)
      builder.setInsertionPointAfter(packedUB.getDefiningOp());
    else {
      builder.setInsertionPoint(moveAnchor);
      if (moveAnchor == packAnchor)
        builder.setInsertionPointAfter(packedUB.getDefiningOp());
    }
    Value offset = builder.create<arith::MulIOp>(
        emission.source.getLoc(), subBlock,
        builder.create<arith::ConstantIndexOp>(emission.source.getLoc(), m16));
    SmallVector<OpFoldResult> offsets{builder.getIndexAttr(0), offset,
                                      builder.getIndexAttr(0),
                                      builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{
        builder.getIndexAttr(n16), builder.getIndexAttr(m16),
        builder.getIndexAttr(kNzTileSize), builder.getIndexAttr(kNzTileSize)};
    SmallVector<OpFoldResult> strides(4, builder.getIndexAttr(1));
    auto view = builder.create<memref::SubViewOp>(
        emission.source.getLoc(), emission.l1, offsets, sizes, strides);
    auto copy = builder.create<hivm::CopyOp>(emission.source.getLoc(),
                                             TypeRange{}, packedUB, view);
    setOpEngineTypeAttr(copy, EngineType::VECTOR);
    graph.nodes[emission.transferNode].op = copy;
    if (emission.ready) {
      emission.ready->moveAfter(copy);
    } else {
      builder.setInsertionPointAfter(copy);
      emission.ready =
          emitBlockSignal(builder, emission.source.getLoc(), Resource::UBMove,
                          Resource::Cube, emission.readyFlag, true);
    }
    SmallVector<unsigned> destinations = edgeEnds(
        graph, emission.transferNode, GraphEdgeKind::TransferDestination, true);
    if (destinations.empty())
      return failure();
    Operation *first = graph.nodes[extreme(graph, destinations, false)].op;
    OpBuilder prelude(scopes.cubeLoop);
    auto ndType = MemRefType::get(
        fullType.getShape(), fullType.getElementType(), nullptr,
        cast<MemRefType>(emission.l1.getType()).getMemorySpace());
    auto nd =
        hivm::DataLayoutAttr::get(prelude.getContext(), hivm::DataLayout::ND);
    auto layout = prelude.create<hivm::ConvertLayoutOp>(
        emission.source.getLoc(), ndType, emission.l1, nd, nd,
        DenseI64ArrayAttr::get(prelude.getContext(), fullType.getShape()),
        ValueRange{});
    setOpEngineTypeAttr(layout, EngineType::CUBE);
    auto plainType =
        MemRefType::get(ndType.getShape(), ndType.getElementType());
    auto spaceCast = prelude.create<memref::MemorySpaceCastOp>(
        emission.source.getLoc(), plainType, layout.getResult());
    setOpEngineTypeAttr(spaceCast, EngineType::CUBE);
    OpBuilder consumer(first);
    emitBlockSignal(consumer, emission.source.getLoc(), Resource::UBMove,
                    Resource::Cube, emission.readyFlag, false);
    auto tensor = consumer.create<bufferization::ToTensorOp>(
        emission.source.getLoc(), fullType, spaceCast.getResult(), true, true);
    setOpEngineTypeAttr(tensor, EngineType::CUBE);
    Value cubeSource = scopes.cubeValues.lookup(emission.originalSource);
    if (!cubeSource)
      return failure();
    for (unsigned destination : destinations)
      graph.nodes[destination].op->replaceUsesOfWith(cubeSource, tensor);
  }
  return success();
}
Operation *acquireOperation(const SSAGraph &graph, unsigned interval) {
  const StorageInterval &storage = graph.storageIntervals[interval];
  Operation *operation = graph.nodes[storage.producer].op;
  if (!operation || storage.space != StorageSpace::L0C)
    return operation;
  auto dps = dyn_cast<DestinationStyleOpInterface>(operation);
  return dps && dps.getNumDpsInits() == 1
             ? dps.getDpsInitOperand(0)->get().getDefiningOp()
             : nullptr;
}
LogicalResult emitOwnershipReleases(
    SSAGraph &graph, PhysicalPlan &plan,
    const DenseMap<unsigned, Operation *> &rewrittenConsumers) {
  for (ReuseRelation &relation : plan.reuse) {
    const StorageInterval &previous =
        graph.storageIntervals[relation.previousInterval];
    const StorageInterval &next = graph.storageIntervals[relation.nextInterval];
    Resource from = graph.nodes[previous.consumer].resource;
    Resource to = graph.nodes[next.producer].resource;
    // Same-engine storage reuse stays a memory dependence.  BiShengIR's
    // intra-core solver owns its pipeline event after physical bufferization;
    // spelling it here fragments otherwise fusible Vector/Cube regions.
    if (engine(from) == engine(to))
      continue;
    Operation *release = rewrittenConsumers.lookup(relation.previousInterval);
    if (!release)
      release = graph.nodes[previous.consumer].op;
    if (!release)
      return failure();
    // A block-sync intrinsic is a scalar-scope operation.  When VFRewrite has
    // outlined the final consumer, publish immediately after that scope rather
    // than accidentally outlining the signal into its SIMD row loop.
    Operation *releaseAnchor = release;
    if (auto outlined = release->getParentOfType<scope::ScopeOp>();
        outlined && outlined->hasAttr("vector_mode"))
      releaseAnchor = outlined;
    OpBuilder set(releaseAnchor);
    set.setInsertionPointAfter(releaseAnchor);
    emitOwnershipSignal(set, graph.loop.getLoc(), from, to, relation.flag,
                        true);
  }
  return success();
}

LogicalResult emitOwnershipAcquires(SSAGraph &graph, PhysicalPlan &plan) {
  for (ReuseRelation &relation : plan.reuse) {
    const StorageInterval &previous =
        graph.storageIntervals[relation.previousInterval];
    const StorageInterval &next = graph.storageIntervals[relation.nextInterval];
    Resource from = graph.nodes[previous.consumer].resource;
    Resource to = graph.nodes[next.producer].resource;
    if (engine(from) == engine(to))
      continue;
    Operation *acquire = acquireOperation(graph, relation.nextInterval);
    if (!acquire)
      return failure();
    OpBuilder wait(acquire);
    emitOwnershipSignal(wait, graph.loop.getLoc(), from, to, relation.flag,
                        false);
  }
  return success();
}
void seedLoopWrapCreditsOnce(const PhysicalPlan &plan, const SSAGraph &graph,
                             ScopeResult &scopes) {
  SmallVector<const ReuseRelation *> cube, vector;
  for (const ReuseRelation &relation : plan.reuse) {
    if (!relation.loopWrap)
      continue;
    const StorageInterval &previous =
        graph.storageIntervals[relation.previousInterval];
    const StorageInterval &next = graph.storageIntervals[relation.nextInterval];
    Resource from = graph.nodes[previous.consumer].resource;
    Resource to = graph.nodes[next.producer].resource;
    if (engine(from) == engine(to))
      continue;
    (engine(from) == EngineType::CUBE ? cube : vector).push_back(&relation);
  }
  auto emit = [&](scope::ScopeOp scope,
                  ArrayRef<const ReuseRelation *> relations) {
    if (relations.empty())
      return;
    Block &body = scope.getBodyRegion().front();
    OpBuilder builder(&body, body.begin());
    scf::ForOp outer = scope->getParentOfType<scf::ForOp>();
    std::optional<OpBuilder> guarded;
    if (outer) {
      Value first = builder.create<arith::CmpIOp>(
          scope.getLoc(), arith::CmpIPredicate::eq, outer.getInductionVar(),
          outer.getLowerBound());
      auto condition =
          builder.create<scf::IfOp>(scope.getLoc(), TypeRange{}, first, false);
      guarded.emplace(condition.getThenBodyBuilder());
    }
    OpBuilder &seed = guarded ? *guarded : builder;
    for (const ReuseRelation *relation : relations) {
      const StorageInterval &previous =
          graph.storageIntervals[relation->previousInterval];
      const StorageInterval &next =
          graph.storageIntervals[relation->nextInterval];
      emitOwnershipSignal(
          seed, scope.getLoc(), graph.nodes[previous.consumer].resource,
          graph.nodes[next.producer].resource, relation->flag, true);
    }
  };
  emit(scopes.cubeScope, cube);
  emit(scopes.vectorScope, vector);
}
} // namespace
Value allocateVectorScratchTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType type, StringRef role) {
  return allocateTensor(builder, loc, type, role, EngineType::VECTOR);
}
LogicalResult lowerSSAToBufferAllocation(func::FuncOp function,
                                         SSAGraph &graph) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  auto target = module->getAttrOfType<hacc::TargetAttr>(hacc::TargetAttr::name);
  if (!target)
    return failure();
  StringRef name = target.getTarget().getValue();
  bool supportedTarget = name == "Ascend950PR_9579" ||
                         name == "Ascend950PR_9589" ||
                         name == "Ascend910_9579" || name == "Ascend910_9589";
  if (!supportedTarget || !graph.loop || !graph.aivPartition ||
      graph.aivPartition->parts != 2 || graph.aivPartition->dimension != 0)
    return failure();
  auto plan = buildIntervals(graph);
  if (failed(plan) || failed(allocateIntervals(function, graph, *plan)) ||
      failed(planOwnership(graph, *plan)) ||
      failed(allocateFlags(function, graph, *plan)))
    return failure();
  SlotPool pool(function, graph.loop, graph, *plan);
  if (!pool.succeeded() || failed(bindL0C(graph, *plan, pool)))
    return failure();
  SmallVector<TransferEmission> v2c;
  DenseMap<unsigned, Operation *> vectorAnchors;
  DenseMap<Value, unsigned> c2vReadIntervals;
  for (const TransferPlan &transfer : plan->transfers) {
    if (graph.nodes[transfer.node].transfer == TransferKind::L0CToUB) {
      if (failed(emitC2V(graph, transfer, pool, vectorAnchors,
                         c2vReadIntervals)))
        return failure();
      continue;
    }
    auto emission = emitV2C(graph, transfer, pool);
    if (failed(emission))
      return failure();
    v2c.push_back(std::move(*emission));
  }
  auto scopes = separateScopes(function, graph);
  if (failed(scopes))
    return failure();
  clonePartitionedInitializers(scopes->vectorScope, *graph.aivPartition);
  Value subBlock = createSubBlockId(scopes->vectorScope);
  DenseMap<Value, Operation *> consumerCompletions;
  if (failed(retileValues(scopes->vectorScope, *graph.aivPartition)) ||
      failed(prepareV2CVFRewrites(graph, v2c, vectorAnchors,
                                  consumerCompletions)) ||
      failed(placeC2VReadsAtFirstUse(scopes->vectorLoop)) ||
      failed(
          retileStores(scopes->vectorScope, subBlock, *graph.aivPartition)) ||
      failed(materializeV2C(graph, v2c, *scopes, *graph.aivPartition,
                            vectorAnchors, subBlock)) ||
      failed(stripScope(scopes->cubeScope, EngineType::CUBE)))
    return failure();
  scopes->cubeLoop = trimLoop(scopes->cubeLoop);
  DenseMap<unsigned, Operation *> rewrittenConsumers;
  for (auto [value, interval] : c2vReadIntervals)
    if (Operation *completion = consumerCompletions.lookup(value))
      rewrittenConsumers.try_emplace(interval, completion);
  if (failed(emitOwnershipReleases(graph, *plan, rewrittenConsumers)))
    return failure();
  if (failed(emitOwnershipAcquires(graph, *plan)))
    return failure();
  seedLoopWrapCreditsOnce(*plan, graph, *scopes);
  LDBG("lowered " << plan->transfers.size() << " transfers into "
                  << graph.physicalSlots.size() << " unified slots");
  return success();
}
} // namespace mlir::triton::cv_split
