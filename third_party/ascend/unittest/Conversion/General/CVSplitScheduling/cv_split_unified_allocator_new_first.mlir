// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=ALLOC --implicit-check-not=memref.reinterpret_cast
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=TOTAL
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=MOVE-NO-REUSE
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=C2V-READY
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=V2C-READY
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=NO-REUSE
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=PLACEMENT
// RUN: triton-opt %s --debug-only=cv-split-ssa-buffer-allocation "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// Two unrolled lanes each contain C2V, V2C, and C2V transfers, covering L0C,
// UB, and L1 through the same allocator. Capacity admits a fresh exact-class
// slot for every logical value, so this case introduces no reuse dependency.

// DIAG: [cv-split-buffer] owned capacity: L0C=8192/262144 UB=5120/253952 L1=2048/524288; downstream PlanMemory required
// DIAG: [cv-split-buffer] assigned 6 block events
// DIAG: [cv-split-buffer] lowered 6 transfers into 12 unified slots

// ALLOC: triton_ascend.cv_split_scheduling.applied = 1 : i32
// ALLOC-LABEL: func.func @new_first_all_spaces
// ALLOC-DAG: memref.alloc() {{.*}}memref<32x16xf32, #hivm.address_space<cc>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<32x16xf32, #hivm.address_space<cc>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<16x16xf32, #hivm.address_space<ub>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<16x16xf32, #hivm.address_space<ub>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<1x1x16x16xf16, #hivm.address_space<ub>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<1x1x16x16xf16, #hivm.address_space<ub>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
// ALLOC-DAG: memref.alloc() {{.*}}memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
// ALLOC-NOT: memref.alloc() {{.*}}#hivm.address_space

// TOTAL-LABEL: func.func @new_first_all_spaces
// TOTAL-COUNT-12: memref.alloc() {{.*}}#hivm.address_space
// TOTAL-NOT: memref.alloc() {{.*}}#hivm.address_space

// Cross-engine UB/L1 buffers dominate both scopes.  Same-engine L0C and
// Pack-to-UB buffers are exact typed allocations inside their owning loop.
// PLACEMENT-LABEL: func.func @new_first_all_spaces
// PLACEMENT-NOT: memref.alloc() {{.*}}#hivm.address_space<cc>
// PLACEMENT: scope.scope
// PLACEMENT: scf.for
// PLACEMENT: %[[CC:.*]] = memref.alloc() {{.*}}memref<32x16xf32, #hivm.address_space<cc>>
// PLACEMENT: %[[CCT:.*]] = bufferization.to_tensor %[[CC]] restrict writable
// PLACEMENT: linalg.fill {{.*}}outs(%[[CCT]]
// PLACEMENT: linalg.matmul
// PLACEMENT: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// PLACEMENT-NOT: memref.alloc() {{.*}}memref<1x1x16x16xf16, #hivm.address_space<ub>>
// PLACEMENT: scope.scope
// PLACEMENT: scf.for
// PLACEMENT: %[[PACK:.*]] = memref.alloc() {{.*}}memref<1x1x16x16xf16, #hivm.address_space<ub>>
// PLACEMENT: memref.collapse_shape %[[PACK]]
// PLACEMENT: bufferization.to_tensor {{.*}} restrict writable
// PLACEMENT: linalg.transpose
// PLACEMENT-NOT: bufferization.materialize_in_destination

// Pack-to-UB intervals do not overlap another compatible producer interval.
// MOVE-NO-REUSE-LABEL: func.func @new_first_all_spaces
// MOVE-NO-REUSE-NOT: hivm.hir.{{set|wait}}_flag[<PIPE_MTE3>, <PIPE_V>

// Every logical transfer owns its readiness event. Producer and consumer IDs
// agree even when transfer payloads have the same shape.
// C2V-READY-LABEL: func.func @new_first_all_spaces
// C2V-READY: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V0:[0-9]+]]
// C2V-READY: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V1:[0-9]+]]
// C2V-READY: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V2:[0-9]+]]
// C2V-READY: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V3:[0-9]+]]
// C2V-READY: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V0]]
// C2V-READY: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V1]]
// C2V-READY: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V2]]
// C2V-READY: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[C2V3]]

// The two V2C values carry independent readiness from Vector to Cube.
// V2C-READY-LABEL: func.func @new_first_all_spaces
// V2C-READY: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[V2C0:[0-9]+]]
// V2C-READY: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[V2C1:[0-9]+]]
// V2C-READY: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[V2C0]]
// V2C-READY: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[V2C1]]

// NO-REUSE-LABEL: func.func @new_first_all_spaces
// NO-REUSE-NOT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
// NO-REUSE-NOT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>]
// NO-REUSE-NOT: hivm.hir.set_flag
// NO-REUSE-NOT: hivm.hir.wait_flag
// NO-REUSE: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @new_first_all_spaces(
      %q: tensor<32x16xf16>, %k: tensor<16x16xf16>,
      %v: tensor<16x16xf16>, %score_init: tensor<32x16xf32>,
      %product_init: tensor<32x16xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %score_zero = linalg.fill ins(%zero : f32) outs(%score_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %product_zero = linalg.fill ins(%zero : f32) outs(%product_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %score = linalg.matmul
          ins(%q, %k : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%score_zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %probability = math.exp %score : tensor<32x16xf32>
      %p16 = arith.truncf %probability : tensor<32x16xf32> to tensor<32x16xf16>
      %product = linalg.matmul
          ins(%p16, %v : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%product_zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %sink = math.exp %product : tensor<32x16xf32>
    }
    return
  }
}
