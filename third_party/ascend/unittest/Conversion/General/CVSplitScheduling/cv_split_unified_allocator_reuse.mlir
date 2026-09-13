// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --implicit-check-not=memref.reinterpret_cast
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=FLOW
// RUN: triton-opt %s --debug-only=cv-split-ssa-buffer-allocation "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// A 32x1024xf32 result needs 128 KiB in L0C and 64 KiB in the per-AIV UB.
// Capacity permits three UB slots but only two L0C slots. Later intervals reuse
// the oldest expired compatible slot only after the corresponding space fills.

// DIAG-COUNT-3: [cv-split-buffer] interval {{[0-9]+}} reused oldest free slot
// DIAG: [cv-split-buffer] owned capacity: L0C=262144/262144 UB=196608/253952 L1=0/524288; downstream PlanMemory required
// DIAG: [cv-split-buffer] assigned 5 block events
// DIAG: [cv-split-buffer] lowered 4 transfers into 5 unified slots

// CHECK: triton_ascend.cv_split_scheduling.applied = 1 : i32
// CHECK-LABEL: func.func @capacity_forces_oldest_free
// CHECK-DAG: memref.alloc() {{.*}}memref<32x1024xf32, #hivm.address_space<cc>>
// CHECK-DAG: memref.alloc() {{.*}}memref<32x1024xf32, #hivm.address_space<cc>>
// CHECK-DAG: memref.alloc() {{.*}}memref<16x1024xf32, #hivm.address_space<ub>>
// CHECK-DAG: memref.alloc() {{.*}}memref<16x1024xf32, #hivm.address_space<ub>>
// CHECK-DAG: memref.alloc() {{.*}}memref<16x1024xf32, #hivm.address_space<ub>>
// CHECK-NOT: memref.alloc() {{.*}}#hivm.address_space

// FLOW-LABEL: func.func @capacity_forces_oldest_free
// FLOW: scope.scope
// FLOW: linalg.matmul
// FLOW: hivm.hir.fixpipe {{.*}} outs(%{{.*}} : memref<16x1024xf32, #hivm.address_space<ub>>)
// FLOW: linalg.matmul
// FLOW: hivm.hir.fixpipe {{.*}} outs(%{{.*}} : memref<16x1024xf32, #hivm.address_space<ub>>)
// FLOW: linalg.matmul
// FLOW: hivm.hir.fixpipe {{.*}} outs(%{{.*}} : memref<16x1024xf32, #hivm.address_space<ub>>)
// FLOW: linalg.matmul
// FLOW: hivm.hir.fixpipe {{.*}} outs(%{{.*}} : memref<16x1024xf32, #hivm.address_space<ub>>)
// FLOW-NOT: hivm.hir.fixpipe
// FLOW: scope.scope
// FLOW-COUNT-4: math.exp

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @capacity_forces_oldest_free(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x1024xf16>,
      %init: tensor<32x1024xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %cube = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x1024xf16>)
          outs(%matmul_init : tensor<32x1024xf32>) -> tensor<32x1024xf32>
      %vector = math.exp %cube : tensor<32x1024xf32>
    }
    return
  }
}
