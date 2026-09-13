# CVSplit Scheduling Design

Status: production architecture implemented in this directory. Build, IR,
accuracy, and performance remain separate validation gates. The current
unified-SSA architecture has not yet produced a qualified hardware performance
result.

## 1. Architecture contract

CVSplit is one transactional compiler pass with five internal stages:

```text
PrepareSSA
  -> optional VFRewrite
  -> BuildSSAGraph
  -> ScheduleSSA
  -> SSABufferAndAllocate
```

The following rules define the implementation.

1. One canonical SSA graph carries semantic dependencies, resource ownership,
   virtual movement, schedule order, storage lifetimes, slot assignments, and
   reuse ownership. Scheduling and memory decisions are not reconstructed from
   a second graph.
2. `PrepareSSA`, and then the optional VF rewrite, operate before graph
   construction. All later analysis and decisions use the same graph.
3. The graph represents four independent resources: Cube, FixPipe, Vector, and
   UBMove/MTE3.
4. Pack, FixPipe, and UBMove are virtual graph nodes. No CVSplit-owned physical
   buffer, transfer, scope, set, or wait is emitted until
   `SSABufferAndAllocate` has a complete valid plan.
5. The scheduler starts ready work on a resource as soon as dependencies and
   that resource permit. Greater reverse depth wins same-resource contention;
   source order is the deterministic tie breaker.
6. A publication `set` is emitted immediately after its concrete transfer. Its
   matching `wait` is emitted immediately before the first consumer.
7. One inclusive-lifetime allocator serves L0C, UB, and L1. The address spaces
   differ only in capacity and compatibility constraints.
8. Unsupported effects, recurrences, transfers, layouts, capacity states, or
   event states reject the candidate. The original IR remains unchanged.
9. **No stage outside `VFRewrite.cpp` may recognise a particular computation.**
   The scheduler, the graph builder, the allocator, and the lowering know about
   resources, edges, layouts, lifetimes, slots, and events. They must never test
   for an operation sequence, an attribute that names a computation, or a buffer
   whose shape only makes sense for one algorithm. Section 8 states the handoff
   contract this rule protects.
10. **The VF rewrite is optional in both directions.** Every input that the pass
    accepts with the rewrite enabled must also be accepted with it disabled, and
    a rewrite that declines a candidate must leave that candidate schedulable.
    Enabling an optimisation may never remove a transformation.
11. CVSplit consumes a verified existing MIX input mapping. It does not silently
    reinterpret a pure 56-AIV launch as 28 MIX programs.

## 2. Vocabulary

Every term is used in exactly this sense for the rest of the document.

| Term | Definition | Defined in |
|---|---|---|
| Engine | A physical execution unit of one A5 AI core: one AIC (Cube) and two AIV sub-blocks (Vector). | hardware |
| Resource | A scheduling lane in the model. Exactly four: `Cube`, `FixPipe`, `Vector`, `UBMove`. FixPipe and UBMove move data; they do not compute. | `Pipeline.h` |
| Engine type | The two-way ownership label projected onto the IR as `ssbuffer.core_type`. Cube and FixPipe project to `CUBE`; Vector and UBMove project to `VECTOR`. | `classifyAllOps.cpp` |
| Graph node | One entry of the canonical graph. Kind is `Operation` for a real MLIR operation, `Pack` for virtual Vector-side packing, or `Transfer` for virtual movement. | `Pipeline.h` |
| Origin id | An integer stamped on every pre-unroll body operation and inherited by its unrolled copies, so the same source operation can be recognised in every lane. It is removed from the IR once the graph exists. | `PreCheck.cpp`, `classifyAllOps.cpp` |
| Lane | One of the `U` copies produced by unrolling. Nodes of one lane descend from one copy. | `PreCheck.cpp` |
| Source order | A deterministic integer per node. Real operations receive 0, 4, 8, ... in body order. A Pack node receives its producer's value plus one and a Transfer node plus one or plus two, so virtual nodes sort immediately after their producer and never collide. | `classifyAllOps.cpp` |
| Level | The logical issue slot assigned by `ScheduleSSA`. A level is not a hardware cycle and carries no duration. | `DependencyScheduler.cpp` |
| Leaf | A node with no zero-distance successor. | `DependencyScheduler.cpp` |
| Reverse depth | The longest remaining edge-count path from a node to any leaf. | section 9 |
| Publication | The concrete movement of a value into another engine's address space, together with its `set` event. | section 12 |
| Ownership | The right to write a physical slot. It is transferred by release and acquire events when two lifetimes share one slot. | section 12 |
| Credit | The initial ownership token seeded once before the loop for a distance-one reuse relation. | section 12 |

### 2.1 Edge kinds

The graph has exactly five edge kinds and one modifier. "SSA edge" in this
document always means the first row.

| Kind | Created by | Meaning | Counts for reverse depth |
|---|---|---|---|
| `SSA` | `BuildSSAGraph` | Ordinary MLIR def-use. The consumer reads an operand defined by the producer, including operands captured by a nested region. It is also used, carrying a memref value, for the represented-memory edge from an admitted `memref.copy` writer to each `bufferization.to_tensor` reader of the same buffer. Those two operations are both users of the allocation and have no def-use edge between them, so the edge is stated explicitly. | yes |
| `TransferSource` | `addVirtualTransfers` | The value leaves its producing resource and enters a virtual Transfer node. | yes |
| `TransferDestination` | `addVirtualTransfers` | The virtual Transfer node delivers the value to a consumer on the other resource. | yes |
| `ResourceOrder` | `closeLocalUsesBeforePack`, then `finalizeOrder` | Program order on one resource. The edges added before scheduling constrain it. The edges added by `finalizeOrder` record its result. | before scheduling: yes; after: no |
| `Ownership` | `SSABufferAndAllocate` | The consumer may not define into a physical slot before the producer's last use of that slot has released it. | no |
| modifier `iterationDistance` | any | Distance in loop iterations. An edge with distance greater than zero is recorded in the edge list but is deliberately not linked into the predecessor and successor lists. This is why a leaf is exactly a node with no successors, and why a loop-wrap edge can never create a cycle in the scheduling DAG. | no |

There is no separate enum value for the represented-memory edge. It is an `SSA`
edge whose value is a memref. That is the only case where an `SSA` edge does not
mean def-use.

## 3. Transaction and stages

The pass clones the input module, runs all stages on the clone, verifies the
result, and commits only a complete valid candidate.

| Stage | Responsibility | Result |
|---|---|---|
| `PrepareSSA` | Select one supported innermost loop, prove structural preconditions, unroll by U2/U4/U8, and attach stable origin identity. | Unrolled structured SSA; no physical CVSplit artifacts. |
| optional `VFRewrite` | Normalise destination-style accumulators, and, when enabled, apply locally proven shaped-expression rewrites. | Equivalent SSA expressions; no scheduling or allocation choice. |
| `BuildSSAGraph` | Classify real operations, build def-use and represented-memory edges, add virtual movement nodes, validate recurrences, and validate the supported two-AIV leading-row partition. | One acyclic canonical graph. |
| `ScheduleSSA` | Compute reverse depth, schedule ready nodes on four resources at their earliest legal levels, and record resource order. | The same graph with levels and order edges. |
| `SSABufferAndAllocate` | Derive inclusive lifetimes, assign physical slots and exact events, then lower virtual movement and form physical scopes. | Verified Cube/Vector MIX IR. |

An inapplicable clone is discarded. A completed candidate that violates an IR
invariant is a compiler error rather than a partial transformation.

## 4. Admission

The pass is an allowlist. Every unsupported case has the same effect: the stage
returns failure, the clone is discarded, and the input module is unchanged.

### 4.1 Loop selection

| Requirement | Why |
|---|---|
| Unroll factor is 2, 4 or 8 | no other factor is modelled |
| Exactly one innermost loop in the function | the pass schedules one body |
| Constant bounds and positive step, so the trip count is static | levels and lifetimes are static |
| Trip count greater than `U` and divisible by `U` | no prologue or epilogue is generated, so a remainder is not representable |
| No nested region-carrying operation and no block successors in the body | control flow inside the body is not modelled |
| Every body value, loop operand and loop result is statically shaped | dynamic geometry has no lifetime and no slot. One exception: a `memref.reinterpret_cast` may consume a dynamically shaped source when its own result is static |
| No memory effect other than Read, except the represented local load below | a store or an unknown effect would require cross-iteration memory analysis the pass does not have |
| Unrolling produces no epilogue loop and the function still verifies | the later stages assume the main loop is the whole loop |

### 4.2 Represented local loads, the one admitted write

An allocation and copy pair in the body is admitted as a represented local load
when all of the following hold:

1. the allocation is a `memref.alloc` in this body with no dynamic sizes;
2. exactly one `memref.copy` in this body writes it, and its source is not the
   buffer itself;
3. every other use is a `bufferization.to_tensor` in this body, and there is at
   least one;
4. no use of any other kind exists;
5. the copy source traces to a function entry argument through view-like
   operations and loop init arguments;
6. the writer precedes every reader in the block.

Conditions 2 to 4 are what make the writer-to-reader edge sound, and that edge
is required, not decorative: `finalizeOrder` physically moves every real
operation into scheduled order, and the copy and its readers have no def-use
edge between them. Condition 5 is conservative narrowing rather than soundness.
This rule proves the local destination's write-before-read relation only. It
does not prove source-pointer stability or cross-iteration aliasing.

### 4.3 Classification

- `linalg.matmul` is Cube.
- The transitive feed closure of a matmul's DPS inputs moves to Cube when every
  operation in that closure is on the feed allowlist: scalar-only operations,
  `bufferization.to_tensor`, `memref.alloc`, `memref.alloca`, `memref.copy`,
  `memref.memory_space_cast`, `tensor.empty`, `linalg.transpose`,
  `tensor.reshape`, `tensor.expand_shape`, `tensor.collapse_shape`, any
  `ViewLikeOpInterface` operation, and `hivm.hir.convert_layout`.
- A matmul accumulator initialised by an in-body constant moves that constant to
  Cube.
- Everything else is Vector. This is a default, not a proof. The admission gates
  in section 4.1 are what keep unknown operations out of the body.

### 4.4 Rejected structures

| Rejected | Reason |
|---|---|
| A loop-carried value whose producer and consumers are not all on one engine | a cross-engine recurrence would need a transfer inside the carried chain |
| A shaped non-matmul Cube value used by anything that is not Cube | the value would need two owners |
| One shaped loop-external input shared by both engines | the input would need two owners |
| A body with no Cube operation or no Vector operation | there is nothing to split |
| A cross-resource edge that is not one of the two supported handoffs | section 8 |
| A graph with no Cube-to-Vector transfer | no row partition can be derived |
| Cube-to-Vector tiles that differ in shape, or whose leading extent is odd | the two-AIV split would not be equal |

## 5. Canonical SSA graph

Each node has one resource, stable source order, predecessors, successors,
reverse depth, and scheduled level. The graph contains real Cube and Vector
operations, virtual Vector Pack nodes, virtual FixPipe nodes, virtual
UBMove/MTE3 nodes, and the edges of section 2.1.

Classification is part of `BuildSSAGraph`. It is not a separate scheduling or
buffer-analysis pipeline.

```text
buildSSAGraph(loop):
  createRawGraph              one node per real operation, source order 0,4,8,...
                              def-use edges including operands captured by nested regions
                              writer-to-reader edges for represented local loads
  classifyMatrixFeeds         matmul and its allowlisted feed closure to Cube
  classifyCarriedState        every loop-carried value stays on one engine
  rejectDualOwnerChains       a shaped non-matmul Cube value is used only by Cube
  validateExternalShapedInputs   one external shaped input serves one engine
  addVirtualTransfers         rebuild every edge; insert Pack and Transfer nodes
  proveAIVPartition           derive the row split and validate the retiled region
  require at least one Cube operation and one Vector operation
  project engine attributes onto the IR and drop origin ids
```

`addVirtualTransfers` clears and rebuilds every predecessor and successor list,
so an edge added before it survives only if it is re-added there.

## 6. Cube-to-Vector handoff

```text
Cube computation -> L0C value -> virtual FixPipe -> UB value -> Vector consumer(s)
```

The producer is a matrix result. Physical lowering binds it to L0C and publishes
it through row-split FixPipe into the private UB halves of the two AIVs.

The L0C lifetime begins at the Cube definition and includes the FixPipe read.
The UB lifetime begins at the same transfer level and ends at the final Vector
use. Because both intervals include the transfer level, source and destination
storage cannot alias at that level.

## 7. Vector-to-Cube handoff

```text
Vector computation -> UB value -> virtual Pack -> virtual UBMove/MTE3
                   -> L1 value -> Cube consumer(s)
```

The value is packed on Vector, copied from UB to L1 by UBMove/MTE3, and consumed
as a matrix input. The UB lifetime includes the copy read. The L1 lifetime
begins at the copy level and ends at the final Cube use.

Pack is a **Vector** node. It occupies a Vector issue slot like any other Vector
operation, and section 10 depends on that.

## 8. The handoff contract

Rule 9 of section 1 is enforced by keeping one narrow interface between the
graph stages and the optional rewrite. These are the rules.

1. **A handoff is described by resource, layout, slot and event. Nothing else.**
   A publication offered to a rewrite carries the published value, the readiness
   signal already emitted for it, the tensor bound to the planned slot, and two
   result fields. It carries no field whose name or shape refers to a
   computation.
2. **The graph stages choose storage. The rewrite chooses computation.** Address
   space, size, alignment, slot identity, reuse and events are allocator
   decisions and are never offered to a rewrite. Scratch storage whose shape
   exists only because of a particular algebraic form is allocated by the
   rewrite, through a generic allocation entry point that does not name what the
   storage is for.
3. **Recognition is asked, never answered, outside the rewrite.** A graph stage
   may ask whether the optional rewrite claims a group of publications. It may
   not test for an operation sequence or for an attribute that names a
   computation in order to answer that question itself.
4. **A rewrite may only replace the producing expression.** What it produces must
   satisfy the layout the handoff already fixed. The destination layout for a
   Vector-to-Cube publication is the packed form the Cube consumer requires, and
   a rewrite that returns anything else is a failure, not a fallback.
5. **A rewrite may position its own operations, including moving the storage it
   writes.** A rewrite can consume a destination earlier than the original
   producer did, so the planned storage is hoisted to the top of the loop body.
   The allocation has no operands, which makes the hoist always legal, and it
   changes no space, size, slot or event decision.
6. **Declining is normal.** A rewrite that is disabled claims nothing. A rewrite
   that is enabled and claims a group may still decline it, and the group is then
   published unchanged through the ordinary path. Only a partially rewritten
   group is an error. This is rule 10 of section 1 in concrete form.
7. **The reverse direction carries only facts the graph stages can act on.** A
   rewrite reports the input whose final consumption it completed, and the
   operation that completes it, so that synchronisation can be placed. It reports
   nothing else.

Destination-style accumulator normalisation is unconditional and is not part of
the optional rewrite, even though it currently lives in the same file. It runs
whether patterns are enabled or not.

### 8.1 Unsupported handoffs

Any cross-resource edge that is not one of the two paths above is rejected,
including an unsupported address space, transfer unit, layout, packing, alias
relation, or a repeated publication of one value with two transfer kinds. The
pass does not guess a transfer and does not let downstream bufferization invent
one. The allowlist can be extended later with new graph-edge semantics; the
fallback remains the original IR until the new edge has complete lifetime,
movement, synchronisation, and verification support.

## 9. `ScheduleSSA`

A leaf is a canonical graph node with no zero-distance successor. `scf.yield` is
structural and is not a graph node. Virtual Pack, FixPipe and UBMove nodes and
the local-use-before-Pack constraints participate in this DAG. Resource-order
edges recorded after scheduling and storage-ownership edges do not.

```text
scheduleSSAGraph(graph):
  closeLocalUsesBeforePack    add ResourceOrder edges so that same-engine uses of a
                              value on the chain feeding a Pack precede that Pack;
                              fail if such an edge would close a cycle
  computeReverseDepth
  assignLevels
  finalizeOrder
```

### 9.1 Reverse depth

```text
remaining[n] := number of successors of n
ready        := every node with no successors          leaves, depth 0
while ready is not empty:
   n := pop(ready)
   for each predecessor p of n:
      depth[p] := max(depth[p], depth[n] + 1)
      if --remaining[p] == 0: push(ready, p)
fail if some node was never popped                     the graph contained a cycle
```

**A node with several leaves takes the maximum, never the sum, the average or the
number of leaves.** For remaining path lengths 3, 5 and 2 the reverse depth is 5.
The relaxation is exact because a node is only pushed once every successor has
reached its final depth.

Reverse depth counts edges. It is not a latency estimate, and a Cube matmul, a
FixPipe move, a Vector elementwise operation and an MTE3 copy all contribute one.
A long chain of cheap Vector operations therefore outranks a short chain that
contains an expensive matmul.

### 9.2 Level assignment

```text
level := 0
while some node is unscheduled:
   for each resource: pick the single highest-priority ready node on it
      priority: greater reverse depth, then smaller source order, then smaller index
   assign level to each picked node                    at most four nodes per level
   only then decrement the predecessor counts of their successors
   level := level + 1
fail if a round picks nothing while work remains
```

Readiness is updated after the whole row is chosen, so a producer and its
consumer never share a level, while independent work on different resources may.

### 9.3 Order

`finalizeOrder` sorts each resource's nodes by level and source order, chains
them with `ResourceOrder` edges, and then physically moves every real operation
before the loop terminator in that order. This is the step that makes section 4.2
necessary.

## 10. Steady state

Levels describe program order inside one body. They are issue slots, not cycles.
For unroll factor 4 with the per-lane chain of sections 6 and 7:

```text
C0[k] -> F0[k] -> V0[k] -> Pack[k] -> U0[k] -> C1[k] -> F1[k] -> V1[k]
```

the per-body node counts are Cube 8, FixPipe 8, **Vector 12**, UBMove 4, because
Pack is a Vector node. Running the algorithm of section 9 on that body, with one
abstract Vector operation per lane, gives:

| Level | Cube | FixPipe | Vector | UBMove |
|---:|---|---|---|---|
| 0 | `C0[0]` | | | |
| 1 | `C0[1]` | `F0[0]` | | |
| 2 | `C0[2]` | `F0[1]` | `V0[0]` | |
| 3 | `C0[3]` | `F0[2]` | `V0[1]` | |
| 4 | | `F0[3]` | `V0[2]` | |
| 5 | | | `V0[3]` | |
| 6 | | | `Pack[0]` | |
| 7 | | | `Pack[1]` | `U0[0]` |
| 8 | `C1[0]` | | `Pack[2]` | `U0[1]` |
| 9 | `C1[1]` | `F1[0]` | `Pack[3]` | `U0[2]` |
| 10 | `C1[2]` | `F1[1]` | `V1[0]` | `U0[3]` |
| 11 | `C1[3]` | `F1[2]` | `V1[1]` | |
| 12 | | `F1[3]` | `V1[2]` | |
| 13 | | | `V1[3]` | |

The body spans 14 levels. The initiation interval is bounded below by the busiest
resource, so it is at least 12, set by Vector and not by Cube. The next
iteration's first Cube node therefore cannot issue before level 12 in this model.
With a realistic six-operation Vector chain per lane the Vector count becomes 32
and the bound becomes 32: such a body is Vector-bound, and adding Cube
concurrency cannot shorten it.

**What actually produces cross-iteration overlap.** After lowering, Cube and
Vector own separate loops in separate scopes. No level table governs the
hardware. Overlap is bounded by two things only:

1. the number of physical slots the allocator created for a compatibility class.
   A class that reserved two slots lets its producer run one value ahead. A class
   that fell back to one reused slot does not; and
2. the loop-wrap ownership credit, which is what allows iteration `g+1` to begin
   writing a slot that iteration `g` last read.

### 10.1 Lifetimes and reuse

Inclusive interval `[definition level, last-use level]` for the schedule above,
with the slot assigned by the rule of section 11. Colour by address space: L0C
red, UB blue, L1 green. Draw reuse as a dashed edge from an interval's last use
to the next definition in the same slot, and label the distance-one loop-wrap
edge.

| Space | Value | Begin | End | Slot | Reused |
|---|---|---:|---:|---|---|
| L0C | first matrix tile, lanes 0..3 | 0,1,2,3 | 1,2,3,4 | s0, s1, s0, s1 | lanes 2,3 |
| L0C | second matrix tile, lanes 0..3 | 8,9,10,11 | 9,10,11,12 | s0, s1, s0, s1 | all |
| UB | Vector result, lanes 0..3 | 1,2,3,4 | 6,7,8,9 | s0, s1, s2, s3 | none |
| UB | packed value, lanes 0..3 | 6,7,8,9 | 7,8,9,10 | s4, s0, s1, s0 | lanes 1..3 |
| UB | second Vector input, lanes 0..3 | 9,10,11,12 | 10,11,12,13 | s4, s2, s1, s3 | all |
| L1 | packed value, lanes 0..3 | 7,8,9,10 | 8,9,10,11 | s0, s1, s0, s1 | lanes 2,3 |

L0C needs two slots, UB five, L1 two. Two intervals in one slot never share a
level, which is the visual form of the inclusive-lifetime rule: a bar must end
strictly before the next bar in the same slot begins.

## 11. Inclusive lifetimes and unified allocation

Each transported value receives an inclusive interval `[definition level,
last-use level]`. Two intervals overlap when one begins at the level where the
other ends. This models a transfer reading its source while writing its
destination.

| Value segment | Space | Inclusive lifetime |
|---|---|---|
| Cube definition to FixPipe completion or later direct Cube use | L0C | Cube definition through final L0C consumer |
| FixPipe publication to final Vector use | UB | FixPipe level through final Vector consumer |
| Vector or Pack definition to UBMove completion | UB | Vector definition through MTE3 read completion |
| UBMove publication to final Cube use | L1 | MTE3 level through final Cube consumer |

Capacities and alignments are fixed constants: L0C 256 KiB at 64 bytes, UB 248
KiB at 32 bytes because policy reserves 8 KiB of the physical 256 KiB, and L1
512 KiB at 32 bytes.

A compatibility class groups intervals that could share one slot: same space,
size, alignment, element type, shape, family and remoteness. Slots are never
shared across classes.

```text
allocateIntervals:
  used[space] := peak bytes already live in the function's own allocations
  order intervals by (begin, end, index)
  group them into compatibility classes
  for each class:
     minimumSlots := 2 when two consecutive members overlap, that is
                       previous.end >= next.begin, or when the schedule has no
                       path from the previous consumer to the next producer
                     else 1
  reject when used plus the sum of class bytes times minimumSlots exceeds a capacity
  for each interval in order:
     if the class has not reached its minimum slot count: create a slot
     else if creating one more still leaves room for every reserved slot: create it
     else reuse the compatible slot whose previous interval ended earliest,
          requiring previous.end < interval.begin
     else reject
  for each remote slot holding more than one interval:
     record a loop-wrap reuse relation from its last interval to its first
```

Two consequences are worth stating plainly. The allocator prefers more slots over
reuse whenever capacity allows, so buffer count is a capacity outcome and not a
user policy. And `minimumSlots = 2` is exactly the double-buffering decision; it
is derived per compatibility class from the schedule and is never requested.

Original loop-local load materialisations participate in the same capacity
decision. A graph-proven allocation, copy and read chain contributes its
scheduled peak live bytes. An allocation with ambiguous users or lifetime
contributes its full size for the complete region.

## 12. Exact synchronisation

Synchronisation is derived after the schedule and slot plan are complete. The
input function must contain no event operation of its own; otherwise the
candidate is rejected. A5 exposes sixteen block-event ids.

| Event | Count | Bound to | Rejection |
|---|---|---|---|
| readiness | one per concrete transfer | the destination interval, which must be remote and must reach its consumer in the graph | more than sixteen events in total |
| ownership | one per reused remote slot, shared by every transition on that slot | the slot | the same budget |
| same-engine reuse | none | ordering only | not applicable |

Readiness has exact endpoints. Cube-to-Vector movement completes on FixPipe and
then sets; Vector executes its wait immediately before the first consumer.
Vector-to-Cube movement completes on UBMove/MTE3 and then sets; Cube waits
immediately before the first consumer. Later users of the same materialisation do
not receive redundant readiness waits.

A non-wrap reuse is legal only when the schedule proves a path from the previous
consumer to the next definition. A loop-wrap reuse is legal only when a
distance-one ownership edge for that exact pair exists; its credit is seeded once
before the loop and then passes from the final use of one iteration to the first
definition of the next.

Ownership may cross resources only in these directions: within one resource,
Vector to FixPipe, Cube to UBMove, FixPipe to Cube, and UBMove to Vector. Any
other transition rejects the candidate rather than emitting an event whose
semantics are unproven.

## 13. Final lowering and MIX mapping

Only `SSABufferAndAllocate` emits physical artifacts. It allocates the selected
L0C, UB and L1 slots, binds matrix destinations to planned L0C slots, lowers
virtual FixPipe and UBMove nodes to concrete movements, replaces supported
cross-resource uses with tensors backed by planned slots, emits exact readiness
and ownership synchronisation, creates one Cube scope and one Vector scope, maps
the validated leading-row halves to the two AIV sub-blocks, and verifies the
module before commit.

### 13.1 What the row partition proves, and what it assumes

The partition is derived from the Cube-to-Vector tiles: rank two, static, all
tiles the same shape, leading extent even, split into two equal halves on
dimension 0. Every Vector operation that touches those rows must be
partition-preserving, and every post-loop store must write a dense static row
view.

Lowering then builds each AIV's view at
`old_offset + sub_block_id * (rows / 2) * row_stride`, with the size on the split
dimension reduced to half.

What is **proven**: the leading extent is even and identical across tiles; the
destination view is static, dense and row-major, so a half-row range is one
contiguous byte range; every Vector operation between the transfer and the store
preserves leading rows; the two views are built from one validated base.

What is **assumed**: that the sub-block id is 0 or 1, that is, that the MIX group
really provides exactly two AIV sub-blocks. Nothing in this pass checks it, and
with a third sub-block the arithmetic above would address beyond the tile.

The two halves are therefore **disjoint by construction under the two-sub-block
MIX contract**, not disjoint by analysis. This pass performs no general region or
alias analysis of AIV address expressions, and no check here inspects an offset.

## 14. Optional VF rewrite boundary

`VFRewritePatterns.td` generates the algebraic patterns. `VFRewrite.cpp` holds
the group materialiser, and it is the only file permitted to recognise a
particular computation. It never chooses the unroll factor, the resource owner,
the schedule, the transfer, the slot, the event or the launch mapping.

With patterns disabled, no publication is claimed and the ordinary path runs.
With patterns enabled, a claimed group that the materialiser declines is also
published through the ordinary path. Section 8 states the full contract.

## 15. Implementation map

| File | Responsibility |
|---|---|
| `CVSplitScheduling.cpp` | Clone, run the five stages, verify, and commit. |
| `PreCheck.cpp` | `PrepareSSA`: admission, origin identity, and unrolling. |
| `VFRewritePatterns.td`, `VFRewrite.cpp` | Accumulator normalisation, and the optional generated rewrites. The only computation-aware code in the pass. |
| `classifyAllOps.cpp` | `BuildSSAGraph`: classification, graph construction, transfer allowlist, recurrence gate, and row-partition validation. |
| `DependencyScheduler.cpp` | `ScheduleSSA`: reverse depth, resource-ASAP scheduling, and order edges. |
| `SSAToBufferAllocation.cpp`, `BufferSlotPlan.h` | `SSABufferAndAllocate`: intervals, slots, events, concrete movement, partitioning, and scopes. |
| `Pipeline.h` | Canonical graph, handoff contract, and stage interfaces. |

## 16. Validation and performance evidence

Qualification requires, in order:

1. a clean build and positive and fallback IR tests, with the optional rewrite
   both enabled and disabled;
2. all 11 established FA shapes accuracy-gated with isolated caches; and
3. a per-shape matched AscendC comparison for all 11 shapes using identical
   source, harness, toolchain, card, launch mapping, profiler and cache policy,
   preferably in an A/B/A bracket. An aggregate is published only after all 11
   rows pass these gates.

The preserved `cvss_4_post_split_materialize` experiment is historical context,
not current unified-SSA evidence:

| Result | Meaning |
|---|---|
| `89.299%` of AscendC | Historical arithmetic combining a 2026-09-07 card-7 CVSplit run with a 2026-08-30 card-3 AscendC run. It is an unmatched mixed-card and mixed-date observation. |
| `81.314502%` of AscendC | Authoritative fresh matched average-time ratio for HD64/CTX32768 on card 0. The named post-split materializer fell back for this shape, so this is not evidence that materialisation improved the kernel. |
| Current unified-SSA architecture | Not yet measured on qualified hardware. No percentage is claimed. |
