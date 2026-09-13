// RUN: triton-opt --split-input-file %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s

// Unsupported transfer topology and unproved layout changes both reject the
// candidate transactionally. No partial unroll or classifier tags may escape.

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @vector_to_cube_through_transpose
// CHECK: %[[ONE0:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE0]] {
// CHECK: linalg.matmul
// CHECK: math.exp
// CHECK: arith.truncf
// CHECK: linalg.transpose
// CHECK: linalg.matmul
// CHECK-NOT: scope.scope
// CHECK-NOT: ssbuffer.core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @vector_to_cube_through_transpose(
      %q: tensor<32x16xf16>, %k: tensor<16x16xf16>,
      %score_init: tensor<32x16xf32>,
      %product_init: tensor<32x32xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %score_zero = linalg.fill ins(%zero : f32) outs(%score_init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %product_zero = linalg.fill ins(%zero : f32) outs(%product_init : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.for %iv = %c0 to %c16 step %c1 {
      %score = linalg.matmul
          ins(%q, %k : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%score_zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %probability = math.exp %score : tensor<32x16xf32>
      %p16 = arith.truncf %probability : tensor<32x16xf32> to tensor<32x16xf16>
      %empty = tensor.empty() : tensor<16x32xf16>
      %transposed = linalg.transpose
          ins(%p16 : tensor<32x16xf16>)
          outs(%empty : tensor<16x32xf16>) permutation = [1, 0]
      %product = linalg.matmul
          ins(%q, %transposed : tensor<32x16xf16>, tensor<16x32xf16>)
          outs(%product_zero : tensor<32x32xf32>) -> tensor<32x32xf32>
    }
    return
  }
}

// -----

// One shaped loop input cannot be shared implicitly by Cube and Vector.
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @shared_shaped_input
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: math.exp %{{.*}} : tensor<32x16xf16>
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @shared_shaped_input(
      %shared: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c16 step %c1 {
      %score = linalg.matmul
          ins(%shared, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%matmul_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %used_by_vector = math.exp %shared : tensor<32x16xf16>
      %result = math.exp %score : tensor<32x16xf32>
    }
    return
  }
}

// -----

// Row splitting stops at an unproved post-loop reshape.
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @unsafe_vector_epilogue
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK: tensor.expand_shape %[[RESULT]]
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsafe_vector_epilogue(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %score_init: tensor<32x16xf32>)
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %state_empty = tensor.empty() : tensor<32x16xf32>
    %state_init = linalg.fill ins(%zero : f32)
        outs(%state_empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %score_empty = tensor.empty() : tensor<32x16xf32>
    %score_zero = linalg.fill ins(%zero : f32)
        outs(%score_empty : tensor<32x16xf32>) -> tensor<32x16xf32>
    %result = scf.for %iv = %c0 to %c16 step %c1
        iter_args(%state = %state_init) -> tensor<32x16xf32> {
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%score_zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %probability = math.exp %score : tensor<32x16xf32>
      %next = arith.addf %state, %probability : tensor<32x16xf32>
      scf.yield %next : tensor<32x16xf32>
    }
    %expanded = tensor.expand_shape %result [[0], [1, 2]]
        output_shape [32, 4, 4] : tensor<32x16xf32> into tensor<32x4x4xf32>
    return
  }
}

// -----

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @unproved_layout_change
// CHECK: %[[ONE1:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE1]] {
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape
// CHECK-NOT: scope.scope
// CHECK-NOT: ssbuffer.core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unproved_layout_change(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c16 step %c1 {
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%matmul_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %expanded = tensor.expand_shape %score [[0], [1, 2]]
          output_shape [32, 4, 4] : tensor<32x16xf32> into tensor<32x4x4xf32>
    }
    return
  }
}

// -----

// An external destination is not assumed to be zero and cannot be discarded.
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @unproved_external_accumulator
// CHECK: linalg.matmul
// CHECK-SAME: outs(%[[INIT:.*]] : tensor<32x16xf32>)
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unproved_external_accumulator(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %iv = %c0 to %c8 step %c1 {
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %score : tensor<32x16xf32>
    }
    return
  }
}

// -----

// An unpartitioned post-loop write would execute on both AIVs after splitting.
// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @unproved_post_loop_write
// CHECK: scf.for
// CHECK: linalg.matmul
// CHECK: math.exp
// CHECK: memref.store
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unproved_post_loop_write(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>, %dst: memref<1xf32>)
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32)
        outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%matmul_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %score : tensor<32x16xf32>
    }
    memref.store %zero, %dst[%c0] : memref<1xf32>
    return
  }
}
