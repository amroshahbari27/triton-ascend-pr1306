// RUN: triton-opt --split-input-file %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s
// RUN: triton-opt --split-input-file %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// Reserving 128 KiB of L0C leaves room for one 128 KiB interval.  The next
// producer is scheduled at the same level where the first interval ends.
// Inclusive lifetimes therefore overlap and the transaction must roll back.

// DIAG: [cv-split] built and classified canonical graph
// DIAG: [cv-split] scheduled
// DIAG: [cv-split-scheduling]: SSABufferAndAllocate rejected inclusive_same_level_is_not_free
// DIAG: [cv-split] built and classified canonical graph
// DIAG: [cv-split] scheduled
// DIAG: [cv-split-scheduling]: SSABufferAndAllocate rejected capacity_exhaustion_rolls_back

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @inclusive_same_level_is_not_free
// CHECK: %[[ONE0:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE0]] {
// CHECK-COUNT-1: linalg.matmul
// CHECK-COUNT-1: math.exp
// CHECK: }
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @inclusive_same_level_is_not_free(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x1024xf16>,
      %init: tensor<32x1024xf32>) attributes {mix_mode = "mix"} {
    %reserved = memref.alloc() : memref<32768xf32, #hivm.address_space<cc>>
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

// -----

// One transfer interval is larger than the entire L0C.  No slot can be
// created or reused, so the original U1 loop is preserved transactionally.

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @capacity_exhaustion_rolls_back
// CHECK: %[[ONE1:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE1]] {
// CHECK-COUNT-1: linalg.matmul
// CHECK-COUNT-1: math.exp
// CHECK: }
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @capacity_exhaustion_rolls_back(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x4096xf16>,
      %init: tensor<32x4096xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %cube = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x4096xf16>)
          outs(%matmul_init : tensor<32x4096xf32>) -> tensor<32x4096xf32>
      %vector = math.exp %cube : tensor<32x4096xf32>
    }
    return
  }
}
