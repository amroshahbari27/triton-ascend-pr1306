/* Copyright (c) Huawei Technologies Co., Ltd. 2026. SPDX-License-Identifier: MIT */
// Schedules one canonical graph across Cube, FixPipe, Vector, and UB movement.
// Ready-node priority is reverse depth followed by deterministic source order.
#include "ascend/include/CVSplitScheduling/Pipeline.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#define DEBUG_TYPE "cv-split-scheduling"
#define LDBG(MSG) LLVM_DEBUG(llvm::dbgs() << "[cv-split] " << MSG << '\n')
namespace mlir::triton::cv_split {
namespace {
constexpr unsigned kResourceCount = 4;
unsigned resourceIndex(Resource resource) { return static_cast<unsigned>(resource); }
StringRef resourceName(Resource resource) {
  static constexpr StringLiteral names[] = {"Cube", "FixPipe", "Vector", "UBMove"};
  unsigned index = resourceIndex(resource);
  if (index >= kResourceCount) llvm_unreachable("unknown resource");
  return names[index];
}
LogicalResult computeReverseDepth(SSAGraph &graph) {
  SmallVector<unsigned> remainingUsers(graph.nodes.size());
  SmallVector<unsigned> ready;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    node.reverseDepth = 0;
    remainingUsers[index] = node.successors.size();
    if (node.successors.empty()) ready.push_back(index);
  }
  unsigned visited = 0;
  while (!ready.empty()) {
    unsigned node = ready.pop_back_val();
    ++visited;
    for (unsigned predecessor : graph.nodes[node].predecessors) {
      graph.nodes[predecessor].reverseDepth =
          std::max(graph.nodes[predecessor].reverseDepth,
                   graph.nodes[node].reverseDepth + 1);
      if (!--remainingUsers[predecessor]) ready.push_back(predecessor);
    }
  }
  return success(visited == graph.nodes.size());
}
bool higherPriority(const SSAGraph &graph, unsigned candidate,
                    unsigned current) {
  const GraphNode &a = graph.nodes[candidate];
  const GraphNode &b = graph.nodes[current];
  if (a.reverseDepth != b.reverseDepth) return a.reverseDepth > b.reverseDepth;
  if (a.sourceOrder != b.sourceOrder) return a.sourceOrder < b.sourceOrder;
  return candidate < current;
}
bool reaches(const SSAGraph &graph, unsigned from, unsigned target) {
  SmallVector<unsigned> pending{from};
  SmallVector<bool> seen(graph.nodes.size());
  while (!pending.empty()) {
    unsigned node = pending.pop_back_val();
    if (node == target) return true;
    if (seen[node]) continue;
    seen[node] = true;
    llvm::append_range(pending, graph.nodes[node].successors);
  }
  return false;
}
bool carriesShapedSSA(const GraphEdge &edge) {
  return edge.kind == GraphEdgeKind::SSA && edge.value &&
         isa<ShapedType>(edge.value.getType());
}
LogicalResult closeLocalUsesBeforePack(SSAGraph &graph) {
  SmallVector<std::pair<unsigned, unsigned>> constraints;
  for (unsigned pack = 0; pack != graph.nodes.size(); ++pack) {
    if (graph.nodes[pack].kind != GraphNodeKind::Pack) continue;
    unsigned towardPack = pack, cursor = pack;
    while (true) {
      SmallVector<unsigned> inputs;
      for (const GraphEdge &edge : graph.edges)
        if (edge.consumer == cursor && carriesShapedSSA(edge) &&
            graph.nodes[edge.producer].kind == GraphNodeKind::Operation &&
            graph.nodes[edge.producer].resource == graph.nodes[pack].resource &&
            !llvm::is_contained(inputs, edge.producer))
          inputs.push_back(edge.producer);
      if (inputs.size() != 1) break;
      cursor = inputs.front();
      for (const GraphEdge &edge : graph.edges) {
        if (edge.producer != cursor || edge.consumer == towardPack ||
            !carriesShapedSSA(edge))
          continue;
        const GraphNode &consumer = graph.nodes[edge.consumer];
        if (consumer.kind != GraphNodeKind::Operation ||
            consumer.resource != graph.nodes[pack].resource)
          continue;
        if (reaches(graph, pack, edge.consumer)) return failure();
        constraints.emplace_back(edge.consumer, pack);
      }
      towardPack = cursor;
    }
  }
  for (auto [producer, pack] : constraints)
    addGraphEdge(graph, producer, pack, GraphEdgeKind::ResourceOrder);
  return success();
}
LogicalResult assignLevels(SSAGraph &graph) {
  SmallVector<unsigned> remainingPredecessors(graph.nodes.size());
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    node.level = kUnscheduledLevel;
    remainingPredecessors[index] = node.predecessors.size();
  }
  unsigned completed = 0;
  for (unsigned level = 0; completed != graph.nodes.size(); ++level) {
    std::array<std::optional<unsigned>, kResourceCount> selected;
    for (unsigned index = 0; index != graph.nodes.size(); ++index) {
      if (graph.nodes[index].level != kUnscheduledLevel ||
          remainingPredecessors[index]) continue;
      unsigned resource = resourceIndex(graph.nodes[index].resource);
      if (resource >= kResourceCount) return failure();
      if (!selected[resource] ||
          higherPriority(graph, index, *selected[resource])) selected[resource] = index;
    }
    bool madeProgress = false;
    for (std::optional<unsigned> choice : selected) {
      if (!choice) continue;
      madeProgress = true;
      graph.nodes[*choice].level = level;
      ++completed;
    }
    if (!madeProgress) return failure();
    // A producer and its consumer occupy different logical levels.  Updating
    // readiness after selecting the whole row preserves that invariant.
    for (std::optional<unsigned> choice : selected)
      if (choice)
        for (unsigned successor : graph.nodes[*choice].successors)
          --remainingPredecessors[successor];
  }
  return success();
}
bool scheduledBefore(const SSAGraph &graph, unsigned a, unsigned b) {
  return std::pair(graph.nodes[a].level, graph.nodes[a].sourceOrder) <
         std::pair(graph.nodes[b].level, graph.nodes[b].sourceOrder);
}
void finalizeOrder(SSAGraph &graph) {
  std::array<SmallVector<unsigned>, kResourceCount> resources;
  SmallVector<unsigned> operations;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    resources[resourceIndex(node.resource)].push_back(index);
    if (node.kind == GraphNodeKind::Operation) operations.push_back(index);
  }
  for (SmallVector<unsigned> &nodes : resources) {
    if (nodes.size() < 2) continue;
    llvm::stable_sort(nodes, [&](unsigned a, unsigned b) {
      return scheduledBefore(graph, a, b); });
    for (unsigned index = 1; index != nodes.size(); ++index)
      addGraphEdge(graph, nodes[index - 1], nodes[index],
                   GraphEdgeKind::ResourceOrder);
  }
  llvm::stable_sort(operations, [&](unsigned a, unsigned b) {
    return scheduledBefore(graph, a, b);
  });
  for (unsigned index : operations)
    graph.nodes[index].op->moveBefore(graph.loop.getBody()->getTerminator());
}
} // namespace
LogicalResult scheduleSSAGraph(SSAGraph &graph) {
  if (!graph.loop || graph.nodes.empty()) return failure();
  // Consume a value locally before publishing its projected cross-engine form.
  // This closes the producer lifetime without buffering same-engine SSA edges.
  if (failed(closeLocalUsesBeforePack(graph))) return failure();
  if (failed(computeReverseDepth(graph)) || failed(assignLevels(graph)))
    return failure();
  finalizeOrder(graph);
  LDBG("scheduled " << graph.nodes.size() << " graph nodes across four resources");
  LLVM_DEBUG({
    for (auto [index, node] : llvm::enumerate(graph.nodes)) {
      llvm::dbgs() << "[cv-split] node " << index << " level=" << node.level
                   << " depth=" << node.reverseDepth
                   << " resource=" << resourceName(node.resource) << " op=";
      StringRef name = node.kind == GraphNodeKind::Pack ? "virtual-pack"
                                                        : "virtual-transfer";
      llvm::dbgs() << (node.op ? node.op->getName().getStringRef() : name) << '\n';
    }
  });
  return success();
}
} // namespace mlir::triton::cv_split
