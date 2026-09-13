/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Builds and classifies one canonical SSA graph before adding virtual movement.
// Unsupported ownership, recurrence, layout, or transfer shapes preserve fallback.
#include "ascend/include/CVSplitScheduling/Attributes.h"
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#define DEBUG_TYPE "cv-split-scheduling"
#define LDBG(MSG) LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << MSG << '\n')
namespace mlir::triton::cv_split {
namespace {
constexpr StringLiteral kCoreTypeAttr = "ssbuffer.core_type";
EngineType getEngine(const GraphNode &node) {
  return node.resource == Resource::Cube ? EngineType::CUBE : EngineType::VECTOR;
}
bool isStaticTransferValue(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() == 2 &&
         llvm::all_of(type.getShape(), [](int64_t size) { return size > 0; });
}
bool isSupportedCubeInput(Operation *consumer, Value value) {
  auto matmul = dyn_cast<linalg::MatmulOp>(consumer);
  return matmul && llvm::is_contained(matmul.getDpsInputs(), value);
}
bool isSupportedCubeResult(Value value) {
  return isa_and_nonnull<linalg::MatmulOp>(value.getDefiningOp()); }
bool hasOnlyScalarTypes(Operation *op) {
  auto scalar = [](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return !shaped || (shaped.hasRank() && shaped.getRank() == 0);
  };
  return op->getNumRegions() == 0 &&
         llvm::all_of(op->getOperandTypes(), scalar) &&
         llvm::all_of(op->getResultTypes(), scalar);
}
bool isAllowlistedCubeFeed(Operation *op) {
  return hasOnlyScalarTypes(op) ||
         isa<bufferization::ToTensorOp, memref::AllocOp, memref::AllocaOp,
             memref::CopyOp, memref::MemorySpaceCastOp, tensor::EmptyOp,
             linalg::TransposeOp, tensor::ReshapeOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp, ViewLikeOpInterface>(op) ||
         op->getName().getStringRef() == "hivm.hir.convert_layout";
}
bool collectCubeFeed(const SSAGraph &graph, unsigned node,
                     DenseSet<unsigned> &visiting,
                     DenseSet<unsigned> &closure) {
  if (closure.contains(node)) return true;
  if (!visiting.insert(node).second) return false;
  Operation *op = graph.nodes[node].op;
  if (!op || (!isa<linalg::MatmulOp>(op) && !isAllowlistedCubeFeed(op))) {
    visiting.erase(node);
    return false;
  }
  for (unsigned predecessor : graph.nodes[node].predecessors) {
    if (!collectCubeFeed(graph, predecessor, visiting, closure)) {
      visiting.erase(node);
      return false;
    }
  }
  visiting.erase(node);
  closure.insert(node);
  return true;
}
void markCube(SSAGraph &graph, const DenseSet<unsigned> &nodes) {
  for (unsigned node : nodes) graph.nodes[node].resource = Resource::Cube;
}
LogicalResult addLocalMaterializationEdges(SSAGraph &graph) {
  Block *body = graph.loop.getBody();
  for (auto copy : body->getOps<memref::CopyOp>()) {
    Value target = copy.getTarget();
    Operation *allocation = target.getDefiningOp();
    if (!allocation || allocation->getBlock() != body ||
        !isa<memref::AllocOp, memref::AllocaOp>(allocation)) {
      LDBG("rejecting copy whose destination is not a local allocation");
      return failure();
    }
    unsigned writers = 0;
    bool hasReader = false;
    for (Operation *user : target.getUsers()) {
      if (isa<memref::CopyOp>(user)) {
        if (user != copy.getOperation()) return failure();
        ++writers;
      } else if (isa<bufferization::ToTensorOp>(user)) {
        hasReader = true;
      } else if (user != allocation) {
        LDBG("rejecting ambiguous local-buffer use by " << user->getName());
        return failure();
      }
    }
    if (writers != 1 || !hasReader) return failure();
    unsigned copyNode = graph.operationNodes.lookup(copy);
    for (Operation *user : target.getUsers()) {
      if (!isa<bufferization::ToTensorOp>(user)) continue;
      auto reader = graph.operationNodes.find(user);
      if (user->getBlock() != body || reader == graph.operationNodes.end())
        return failure();
      addGraphEdge(graph, copyNode, reader->second, GraphEdgeKind::SSA, target);
    }
  }
  return success();
}
LogicalResult createRawGraph(scf::ForOp loop, SSAGraph &graph) {
  graph.loop = loop;
  Block *body = loop.getBody();
  unsigned sourceOrder = 0;
  for (Operation &op : body->without_terminator()) {
    auto origin = op.getAttrOfType<IntegerAttr>(kOriginAttr);
    if (origin && origin.getInt() < 0)
      return failure();
    int64_t originId = origin ? origin.getInt() : -1;
    unsigned node = graph.nodes.size();
    graph.operationNodes[&op] = node;
    graph.nodes.push_back({&op, {}, GraphNodeKind::Operation, Resource::Vector,
                           TransferKind::None, originId, sourceOrder, 0,
                           kUnscheduledLevel, {}, {}});
    sourceOrder += 4;
  }
  for (Operation &consumer : body->without_terminator()) {
    unsigned consumerNode = graph.operationNodes.lookup(&consumer);
    auto addDependency = [&](Value value) -> LogicalResult {
      Operation *producer = value.getDefiningOp();
      while (producer && producer->getBlock() != body)
        producer = producer->getParentOp();
      if (!producer || producer == &consumer || producer->getBlock() != body)
        return success();
      auto producerNode = graph.operationNodes.find(producer);
      if (producerNode == graph.operationNodes.end())
        return failure();
      bool exists = llvm::any_of(graph.edges, [&](const GraphEdge &edge) {
        return edge.kind == GraphEdgeKind::SSA &&
               edge.producer == producerNode->second &&
               edge.consumer == consumerNode && edge.value == value;
      });
      if (!exists)
        addGraphEdge(graph, producerNode->second, consumerNode,
                     GraphEdgeKind::SSA, value);
      return success();
    };
    for (Value value : consumer.getOperands())
      if (failed(addDependency(value)))
        return failure();
    WalkResult captured = consumer.walk([&](Operation *nested) {
      if (nested == &consumer)
        return WalkResult::advance();
      for (Value value : nested->getOperands())
        if (failed(addDependency(value)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (captured.wasInterrupted())
      return failure();
  }
  return addLocalMaterializationEdges(graph);
}
LogicalResult classifyMatrixFeeds(SSAGraph &graph) {
  SmallVector<linalg::MatmulOp> matmuls;
  for (GraphNode &node : graph.nodes) {
    auto matmul = dyn_cast_or_null<linalg::MatmulOp>(node.op);
    if (!matmul) continue;
    matmuls.push_back(matmul);
    node.resource = Resource::Cube;
  }
  if (matmuls.empty()) return failure();
  for (linalg::MatmulOp matmul : matmuls) {
    for (Value input : matmul.getDpsInputs()) {
      Operation *producer = input.getDefiningOp();
      if (!producer || producer->getBlock() != graph.loop.getBody()) continue;
      DenseSet<unsigned> visiting, closure;
      unsigned node = graph.operationNodes.lookup(producer);
      if (collectCubeFeed(graph, node, visiting, closure)) markCube(graph, closure);
      else LDBG("keeping a computed matrix operand on Vector");
    }
    Value init = matmul.getDpsInits().front();
    if (auto constant = init.getDefiningOp<arith::ConstantOp>();
        constant && constant->getBlock() == graph.loop.getBody())
      graph.nodes[graph.operationNodes.lookup(constant)].resource = Resource::Cube;
  }
  return success();
}
LogicalResult classifyCarriedState(SSAGraph &graph) {
  auto yield = dyn_cast<scf::YieldOp>(graph.loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != graph.loop.getRegionIterArgs().size())
    return failure();
  for (auto [argument, next] :
       llvm::zip(graph.loop.getRegionIterArgs(), yield.getOperands())) {
    if (argument == next) continue;
    Operation *producer = next.getDefiningOp();
    if (!producer || producer->getBlock() != graph.loop.getBody()) {
      LDBG("rejecting carried value without an in-body SSA producer");
      return failure();
    }
    DenseSet<unsigned> visiting, nextClosure;
    unsigned producerNode = graph.operationNodes.lookup(producer);
    bool canMoveToCube = collectCubeFeed(graph, producerNode, visiting, nextClosure);
    std::optional<EngineType> observed;
    SmallVector<unsigned> consumers;
    for (Operation *consumer : argument.getUsers()) {
      if (consumer == yield) continue;
      auto node = graph.operationNodes.find(consumer);
      if (consumer->getBlock() != graph.loop.getBody() ||
          node == graph.operationNodes.end())
        return failure();
      consumers.push_back(node->second);
      if (canMoveToCube && nextClosure.contains(node->second)) continue;
      EngineType owner = getEngine(graph.nodes[node->second]);
      if (observed && *observed != owner) {
        LDBG("rejecting cross-resource loop-carried value");
        return failure();
      }
      observed = owner;
    }
    if (observed == EngineType::CUBE && canMoveToCube) markCube(graph, nextClosure);
    EngineType owner = getEngine(graph.nodes[producerNode]);
    if (llvm::any_of(consumers, [&](unsigned node) {
          return getEngine(graph.nodes[node]) != owner;
        })) {
      LDBG("rejecting cross-resource loop-carried value");
      return failure();
    }
  }
  return success();
}
LogicalResult rejectDualOwnerChains(SSAGraph &graph) {
  for (const GraphNode &node : graph.nodes) {
    if (!node.op || node.resource != Resource::Cube ||
        isa<linalg::MatmulOp>(node.op))
      continue;
    for (Value result : node.op->getResults()) {
      auto shaped = dyn_cast<ShapedType>(result.getType());
      if (!shaped || (shaped.hasRank() && shaped.getRank() == 0))
        continue;
      for (Operation *user : result.getUsers()) {
        if (isa<scf::YieldOp>(user)) continue;
        auto owner = graph.operationNodes.find(user);
        if (user->getBlock() != graph.loop.getBody() ||
            owner == graph.operationNodes.end() ||
            graph.nodes[owner->second].resource != Resource::Cube) {
          LDBG("rejecting shaped load/format chain with two owners");
          return failure();
        }
      }
    }
  }
  return success();
}
LogicalResult validateExternalShapedInputs(SSAGraph &graph) {
  auto yield = cast<scf::YieldOp>(graph.loop.getBody()->getTerminator());
  DenseMap<Value, EngineType> owners;
  for (Operation &consumer : graph.loop.getBody()->without_terminator()) {
    EngineType owner =
        getEngine(graph.nodes[graph.operationNodes.lookup(&consumer)]);
    for (Value value : consumer.getOperands()) {
      auto shaped = dyn_cast<ShapedType>(value.getType());
      Operation *producer = value.getDefiningOp();
      if (!shaped || (shaped.hasRank() && shaped.getRank() == 0) ||
          (producer && producer->getBlock() == graph.loop.getBody()))
        continue;
      Value identity = value;
      if (auto argument = dyn_cast<BlockArgument>(value);
          argument && argument.getOwner() == graph.loop.getBody() &&
          argument.getArgNumber() > 0 &&
          yield.getOperand(argument.getArgNumber() - 1) == argument)
        identity = graph.loop.getInitArgs()[argument.getArgNumber() - 1];
      auto [entry, inserted] = owners.try_emplace(identity, owner);
      if (!inserted && entry->second != owner) {
        LDBG("rejecting shaped loop input shared across engines");
        return failure();
      }
    }
  }
  return success();
}
LogicalResult addVirtualTransfers(SSAGraph &graph) {
  SmallVector<GraphEdge> rawEdges = std::move(graph.edges);
  graph.edges.clear();
  for (GraphNode &node : graph.nodes) {
    node.predecessors.clear();
    node.successors.clear();
  }
  DenseMap<Value, unsigned> transfers, packs;
  for (const GraphEdge &edge : rawEdges) {
    unsigned producer = edge.producer, consumer = edge.consumer;
    Resource from = graph.nodes[producer].resource;
    Resource to = graph.nodes[consumer].resource;
    if (from == to) {
      addGraphEdge(graph, producer, consumer, edge.kind, edge.value,
                   edge.iterationDistance);
      continue;
    }
    if (edge.kind != GraphEdgeKind::SSA || !isStaticTransferValue(edge.value))
      return failure();
    TransferKind kind;
    Resource movement;
    unsigned source = producer;
    if (from == Resource::Cube && to == Resource::Vector &&
        isSupportedCubeResult(edge.value)) {
      kind = TransferKind::L0CToUB;
      movement = Resource::FixPipe;
    } else if (from == Resource::Vector && to == Resource::Cube &&
               isSupportedCubeInput(graph.nodes[consumer].op, edge.value)) {
      kind = TransferKind::UBToL1;
      movement = Resource::UBMove;
      auto [pack, inserted] = packs.try_emplace(edge.value, 0);
      if (inserted) {
        pack->second = graph.nodes.size();
        graph.nodes.push_back({nullptr, edge.value, GraphNodeKind::Pack,
                               Resource::Vector, TransferKind::None,
                               graph.nodes[producer].origin,
                               graph.nodes[producer].sourceOrder + 1, 0,
                               kUnscheduledLevel, {}, {}});
        addGraphEdge(graph, producer, pack->second, GraphEdgeKind::SSA, edge.value);
      }
      source = pack->second;
    } else {
      LDBG("rejecting unsupported cross-resource edge");
      return failure();
    }
    auto [transfer, inserted] = transfers.try_emplace(edge.value, 0);
    if (inserted) {
      transfer->second = graph.nodes.size();
      graph.nodes.push_back({nullptr, edge.value, GraphNodeKind::Transfer,
                             movement, kind, graph.nodes[producer].origin,
                             graph.nodes[producer].sourceOrder +
                             (kind == TransferKind::UBToL1 ? 2 : 1), 0,
                             kUnscheduledLevel, {}, {}});
      addGraphEdge(graph, source, transfer->second,
                   GraphEdgeKind::TransferSource, edge.value);
    } else if (graph.nodes[transfer->second].transfer != kind) {
      return failure();
    }
    addGraphEdge(graph, transfer->second, consumer,
                 GraphEdgeKind::TransferDestination, edge.value);
  }
  return success();
}
bool hasPartitionExtent(Type type, const AIVPartition &partition) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasRank() && shaped.hasStaticShape() &&
         partition.dimension < static_cast<unsigned>(shaped.getRank()) &&
         shaped.getDimSize(partition.dimension) == partition.fullExtent;
}
bool preservesLeadingRows(Type type, const AIVPartition &partition) {
  auto shaped = dyn_cast<ShapedType>(type);
  return !shaped || (shaped.hasRank() && shaped.getRank() == 0) ||
         hasPartitionExtent(type, partition);
}
bool preservesLeadingRows(Operation *op, const AIVPartition &partition) {
  auto preserves = [&](Type type) { return preservesLeadingRows(type, partition); };
  return llvm::all_of(op->getOperandTypes(), preserves) && llvm::all_of(op->getResultTypes(), preserves);
}
bool isStaticDisjointRowView(memref::ReinterpretCastOp op,
                             const AIVPartition &partition) {
  auto type = dyn_cast<MemRefType>(op.getType());
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();
  if (!type || !hasPartitionExtent(type, partition) ||
      sizes.size() != static_cast<size_t>(type.getRank()) ||
      strides.size() != static_cast<size_t>(type.getRank()) ||
      llvm::is_contained(sizes, ShapedType::kDynamic) ||
      llvm::is_contained(strides, ShapedType::kDynamic))
    return false;
  int64_t expected = 1;
  for (unsigned i = type.getRank(); i-- > 0;) {
    if (sizes[i] <= 0 || strides[i] != expected ||
        sizes[i] > std::numeric_limits<int64_t>::max() / expected)
      return false;
    expected *= sizes[i];
  }
  return true;
}
bool touchesPartitionedRows(Operation *op, const AIVPartition &partition) {
  auto touches = [&](Type type) { return hasPartitionExtent(type, partition); };
  return llvm::any_of(op->getOperandTypes(), touches) ||
         llvm::any_of(op->getResultTypes(), touches);
}
bool isPartitionPreservingVectorOp(Operation *op,
                                   const AIVPartition &partition) {
  if (hasOnlyScalarTypes(op)) return true;
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    return dense && dense.isSplat() && preservesLeadingRows(op, partition);
  }
  if (op->hasTrait<OpTrait::Elementwise>()) return preservesLeadingRows(op, partition);
  if (isa<tensor::EmptyOp, linalg::FillOp, memref::AllocOp, memref::AllocaOp>(op))
    return preservesLeadingRows(op, partition);
  if (auto tensor = dyn_cast<bufferization::ToTensorOp>(op)) {
    Operation *allocation = tensor.getBuffer().getDefiningOp();
    return allocation && allocation->getBlock() == op->getBlock() &&
           isa<memref::AllocOp, memref::AllocaOp>(allocation) &&
           preservesLeadingRows(op, partition);
  }
  if (auto reduce = dyn_cast<linalg::ReduceOp>(op))
    return !llvm::is_contained(reduce.getDimensions(), int64_t{0}) && preservesLeadingRows(op, partition);
  if (auto broadcast = dyn_cast<linalg::BroadcastOp>(op))
    return !llvm::is_contained(broadcast.getDimensions(), int64_t{0}) && preservesLeadingRows(op, partition);
  if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    return !permutation.empty() && permutation.front() == 0 &&
           preservesLeadingRows(op, partition);
  }
  if (auto view = dyn_cast<memref::ReinterpretCastOp>(op))
    return isStaticDisjointRowView(view, partition);
  return false;
}
bool isProvenPartitionedStore(Operation *op,
                               const AIVPartition &partition) {
  auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op);
  if (!store || !preservesLeadingRows(op, partition)) return false;
  auto view = store.getDest().getDefiningOp<memref::ReinterpretCastOp>();
  return view && isStaticDisjointRowView(view, partition);
}
bool hasOnlyReadOrNoEffect(Operation *op) {
  if (isMemoryEffectFree(op)) return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects) return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect());
  });
}
bool hasRetileableExternalOperands(Operation *op, scf::ForOp loop,
                                   const AIVPartition &partition) {
  for (OpOperand &use : op->getOpOperands()) {
    if (auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op);
        store && use.get() == store.getDest()) continue;
    Value value = use.get();
    if (!hasPartitionExtent(value.getType(), partition)) continue;
    Operation *definition = value.getDefiningOp();
    if (definition && (definition->getBlock() == loop.getBody() ||
        definition == loop.getOperation() ||
        (definition->getBlock() == loop->getBlock() && loop->isBeforeInBlock(definition))))
      continue;
    if (auto argument = dyn_cast<BlockArgument>(value);
        argument && argument.getOwner() == loop.getBody() &&
        argument.getArgNumber() > 0) {
      value = loop.getInitArgs()[argument.getArgNumber() - 1];
      definition = value.getDefiningOp();
    }
    if (!definition || !isa<tensor::EmptyOp, linalg::FillOp>(definition)) {
      LDBG("rejecting Vector operand without a retileable initializer");
      return false;
    }
  }
  return true;
}
LogicalResult validateRetiledRegion(SSAGraph &graph,
                                    const AIVPartition &partition) {
  for (const GraphNode &node : graph.nodes) {
    if (node.kind != GraphNodeKind::Operation || node.resource != Resource::Vector || !node.op ||
        !touchesPartitionedRows(node.op, partition))
      continue;
    if (!isPartitionPreservingVectorOp(node.op, partition) ||
        !hasRetileableExternalOperands(node.op, graph.loop, partition)) {
      LDBG("rejecting unproved row-partitioned Vector operation " << node.op->getName());
      return failure();
    }
  }
  Block *parent = graph.loop->getBlock();
  bool afterLoop = false;
  for (Operation &op : parent->without_terminator()) {
    if (&op == graph.loop.getOperation()) { afterLoop = true; continue; }
    if (!afterLoop) continue;
    if (isa<linalg::MatmulOp, RegionBranchOpInterface>(&op) ||
        (!isProvenPartitionedStore(&op, partition) &&
         !hasOnlyReadOrNoEffect(&op)) ||
        (touchesPartitionedRows(&op, partition) &&
         ((!isPartitionPreservingVectorOp(&op, partition) &&
           !isProvenPartitionedStore(&op, partition)) ||
          !hasRetileableExternalOperands(&op, graph.loop, partition)))) {
      LDBG("rejecting unproved post-loop Vector operation " << op.getName());
      return failure();
    }
    setOpEngineTypeAttr(&op, EngineType::VECTOR);
  }
  for (Value result : graph.loop.getResults()) {
    for (Operation *user : result.getUsers()) {
      Operation *top = user;
      while (top && top->getBlock() != parent) top = top->getParentOp();
      if (!top || top->hasTrait<OpTrait::IsTerminator>() ||
          !graph.loop->isBeforeInBlock(top)) {
        LDBG("rejecting loop result that escapes the Vector epilogue");
        return failure();
      }
    }
  }
  return success();
}
LogicalResult proveAIVPartition(SSAGraph &graph) {
  std::optional<AIVPartition> partition;
  SmallVector<int64_t> c2vShape;
  for (const GraphNode &node : graph.nodes) {
    if (node.kind != GraphNodeKind::Transfer) continue;
    auto type = dyn_cast<RankedTensorType>(node.value.getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 2) return failure();
    int64_t extent = type.getDimSize(0);
    if (extent <= 0 || extent % 2) return failure();
    if (node.transfer == TransferKind::L0CToUB) {
      if (c2vShape.empty())
        llvm::append_range(c2vShape, type.getShape());
      else if (!llvm::equal(c2vShape, type.getShape())) {
        LDBG("rejecting unqualified asymmetric Cube-to-Vector geometry");
        return failure();
      }
      if (!partition)
        partition = AIVPartition{0, extent, extent / 2, 2};
    }
    if (partition && partition->fullExtent != extent) return failure();
  }
  if (!partition) {
    LDBG("rejecting graph without a provable Cube-to-Vector row partition"); return failure();
  }
  if (failed(validateRetiledRegion(graph, *partition))) return failure();
  graph.aivPartition = *partition;
  return success();
}
void projectClassification(SSAGraph &graph) {
  for (const GraphNode &node : graph.nodes)
    if (node.kind == GraphNodeKind::Operation && node.op) setOpEngineTypeAttr(node.op, getEngine(node));
}
} // namespace
void addGraphEdge(SSAGraph &graph, unsigned producer, unsigned consumer,
                  GraphEdgeKind kind, Value value, unsigned iterationDistance) {
  assert(producer < graph.nodes.size() && consumer < graph.nodes.size());
  graph.edges.push_back({producer, consumer, kind, value, iterationDistance});
  if (iterationDistance) return;
  if (!llvm::is_contained(graph.nodes[producer].successors, consumer))
    graph.nodes[producer].successors.push_back(consumer);
  if (!llvm::is_contained(graph.nodes[consumer].predecessors, producer))
    graph.nodes[consumer].predecessors.push_back(producer);
}
void setOpEngineTypeAttr(Operation *op, EngineType engine) {
  StringRef name = engine == EngineType::CUBE ? "CUBE" : "VECTOR";
  op->setAttr(kCoreTypeAttr, StringAttr::get(op->getContext(), name)); }
void removeEngineTypeAttrs(ModuleOp module) {
  module.walk([](Operation *op) { op->removeAttr(kCoreTypeAttr); }); }
FailureOr<SSAGraph> buildSSAGraph(scf::ForOp loop) {
  if (!loop) return failure();
  SSAGraph graph;
  if (failed(createRawGraph(loop, graph)) || failed(classifyMatrixFeeds(graph)) ||
      failed(classifyCarriedState(graph)) || failed(rejectDualOwnerChains(graph)) ||
      failed(validateExternalShapedInputs(graph)) ||
      failed(addVirtualTransfers(graph)) || failed(proveAIVPartition(graph)))
    return failure();
  bool hasCube = false, hasVector = false;
  for (const GraphNode &node : graph.nodes) {
    if (node.kind != GraphNodeKind::Operation) continue;
    (node.resource == Resource::Cube ? hasCube : hasVector) = true;
  }
  if (!hasCube || !hasVector) return failure();
  projectClassification(graph);
  loop.walk([](Operation *operation) { operation->removeAttr(kOriginAttr); });
  LDBG("built and classified canonical graph with " << graph.nodes.size()
       << " nodes and " << graph.edges.size() << " edges");
  return graph;
}
} // namespace mlir::triton::cv_split
