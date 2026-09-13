// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// CVSplit does not invent launcher semantics. A structurally valid graph
// without a verified MIX mapping is preserved for the fallback pipeline.

// DIAG: input has no verified MIX launch mapping; preserving fallback
// DIAG: candidate was not applicable; preserving the DCVP fallback input

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-NOT: hivm.disable_auto_tile_and_bind_subblock
// CHECK-LABEL: func.func @requires_mix_launch
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE]] {
// CHECK: linalg.matmul
// CHECK: math.exp
// CHECK-NOT: scope.scope
// CHECK-NOT: ssbuffer.core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @requires_mix_launch(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %iv = %c0 to %c8 step %c1 {
      %zero = arith.constant dense<0.0> : tensor<32x16xf32>
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %vector = math.exp %score : tensor<32x16xf32>
    }
    return
  }
}
