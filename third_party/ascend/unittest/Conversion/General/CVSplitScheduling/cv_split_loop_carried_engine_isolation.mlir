// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// A Cube-produced loop result cannot feed a Vector consumer in the next
// original iteration. The canonical SSA graph rejects it before lowering.

// DIAG: [cv-split] rejecting cross-resource loop-carried value
// DIAG: [cv-split-scheduling]: candidate was not applicable; preserving the DCVP fallback input

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-NOT: ssbuffer.core_type
// CHECK-LABEL: func.func @cross_resource_recurrence
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE]] iter_args
// CHECK: %[[VECTOR:.*]] = math.exp
// CHECK: %[[CUBE:.*]] = linalg.matmul
// CHECK: scf.yield %[[CUBE]]
// CHECK: return %[[RESULT]]
// CHECK-NOT: scope.scope

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @cross_resource_recurrence(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) -> tensor<32x16xf32>
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %cube_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    %result = scf.for %iv = %c0 to %c8 step %c1
        iter_args(%state = %init) -> tensor<32x16xf32> {
      %vector = math.exp %state : tensor<32x16xf32>
      %cube = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%cube_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      scf.yield %cube : tensor<32x16xf32>
    }
    return %result : tensor<32x16xf32>
  }
}
