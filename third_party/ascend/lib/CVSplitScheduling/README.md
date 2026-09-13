# CVSplit scheduling

CVSplit is one transactional MLIR pass that schedules a verified MIX kernel
across Cube, FixPipe, Vector, and UBMove/MTE3. It keeps every post-preparation
decision on one canonical SSA graph and commits only a complete verified
candidate.

## Pipeline

```text
PrepareSSA
  -> optional TableGen VFRewrite
  -> BuildSSAGraph
  -> ScheduleSSA
  -> SSABufferAndAllocate
```

| Stage | Purpose |
|---|---|
| `PrepareSSA` | Select a supported innermost loop, prove preconditions, unroll by U2/U4/U8, and preserve operation/lane lineage. |
| optional `VFRewrite` | Apply locally equivalent patterns generated from `VFRewritePatterns.td`. |
| `BuildSSAGraph` | Classify Cube/Vector work, build semantic and represented-memory dependencies, add virtual Pack/FixPipe/UBMove nodes, validate loop recurrence, and prove the two-AIV row split. |
| `ScheduleSSA` | Start ready work on each resource at its earliest legal level. Greater reverse depth wins same-resource contention; source order breaks ties. |
| `SSABufferAndAllocate` | Derive inclusive lifetimes, assign L0C/UB/L1 slots and exact events, then emit concrete movements and Cube/Vector scopes. |

`Pipeline.h` defines the shared graph. No CVSplit-owned physical buffer,
transfer, scope, set, or wait exists before the final stage.

## Supported graph

```text
Cube -> L0C -> FixPipe -> private UB -> Vector

Vector -> private UB -> Pack -> UBMove/MTE3 -> shared L1 -> Cube
```

The graph schedules movement independently from computation. FixPipe may
publish one Cube result while Cube computes another. UBMove/MTE3 may publish a
packed Vector result while later Vector work executes.

The current pass rejects stores, in-place or unknown effects, cross-resource
loop-carried dependencies, unsupported transfer edges, unproved layouts or
aliases, capacity overflow, and event exhaustion. Rejection discards the clone
and leaves the original IR unchanged.

## Scheduling, allocation, and events

Reverse depth is the longest dependency path from a node to a leaf. At each
logical level, the scheduler selects at most one ready node on each of Cube,
FixPipe, Vector, and UBMove/MTE3. It chooses greater reverse depth, then source
order. There are no candidate-specific scheduling modes.

Storage intervals are inclusive: a source being read and a destination being
written at one transfer level overlap. One generic allocator is used for L0C,
UB, and L1:

1. preserve two slots when intervals overlap or cross-resource execution lacks
   an ordering path;
2. then reuse the compatible expired slot whose preceding interval ended
   earliest;
3. create another slot only for a real overlapping lifetime;
4. reject when the required slot exceeds memory or event capacity.

Graph-proven loop-local load allocations contribute their peak scheduled bytes;
ambiguous allocations remain permanent reservations. The address spaces differ
only by capacity and compatibility. Readiness `set` is placed immediately after
concrete movement; its `wait` is placed immediately before the first consumer.
Reusing a physical slot creates an exact ownership relation. Periodic reuse adds
prologue seeding, a loop-wrap release/acquire, and drain consumption.

## MIX contract

The input function must already have a verified MIX launch mapping. CVSplit
does not silently convert a pure 56-AIV grid into 28 MIX programs. For an
accepted candidate, the existing MIX group supplies one AIC and two AIV
subblocks; the graph proof assigns the AIVs disjoint row halves.

## Options

- `compile-on-910-95`
- `unroll-factor` in `{2, 4, 8}`
- `enable-vf-rewrite`

The backend controls are `enable_cv_split_scheduling`,
`cv_split_unroll_factor`, and `cv_split_enable_vf_rewrite`. Scheduling order,
transfer timing, buffer count, reuse, and event placement are graph-derived,
not user-selectable policies.

## Files

| File | Responsibility |
|---|---|
| `CVSplitScheduling.cpp` | Transaction, orchestration, verification, and commit. |
| `PreCheck.cpp` | `PrepareSSA`. |
| `VFRewritePatterns.td`, `VFRewrite.cpp` | Optional TableGen VF rewrites. |
| `classifyAllOps.cpp` | `BuildSSAGraph`, including supported accumulator normalization. |
| `DependencyScheduler.cpp` | `ScheduleSSA`. |
| `SSAToBufferAllocation.cpp`, `BufferSlotPlan.h` | `SSABufferAndAllocate`. |
| `Pipeline.h` | Canonical graph and stage interfaces. |

The active implementation is 1,979 physical lines plus 18 integration lines:
1,997 total, excluding tests, documentation, and generated files.

## Evidence boundary

Build/IR, hardware accuracy, and matched hardware performance are independent
gates. The historical `89.299%` value used CVSplit and AscendC measurements
from different cards and dates. The authoritative matched HD64/CTX32768 replay
was `81.314502%` of AscendC, and the named historical materializer fell back on
that shape. The current unified-SSA architecture is not yet performance
qualified and makes no percentage claim.

See [DESIGN.md](DESIGN.md) for the graph, lifetime, synchronization, steady-state
schedule, fallback, and provenance details.
