// RUN: triton-opt --split-input-file %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s

// The independent Vector chain gives classification both engines but no legal
// cross-resource value. Rejection must preserve the original loop exactly.

// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-NOT: hivm.disable_auto_tile_and_bind_subblock
// CHECK-NOT: ssbuffer.core_type
// CHECK-LABEL: func.func @unsupported_independent_chains
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE]] {
// CHECK: %{{.*}} = linalg.matmul
// CHECK: %{{.*}} = math.exp %{{.*}} : tensor<32x16xf32>
// CHECK: }
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsupported_independent_chains(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>, %other: tensor<32x16xf32>)
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c16 step %c1 {
      %cube = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%matmul_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %other : tensor<32x16xf32>
    }
    return
  }
}

// -----

// A memory write is rejected during preparation, before graph construction.
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @store_in_loop
// CHECK: %[[ONE1:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE1]] {
// CHECK: linalg.matmul
// CHECK: math.exp
// CHECK: memref.store
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @store_in_loop(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>, %dst: memref<16xf32>, %value: f32)
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %iv = %c0 to %c16 step %c1 {
      %cube = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %cube : tensor<32x16xf32>
      memref.store %value, %dst[%iv] : memref<16xf32>
    }
    return
  }
}
