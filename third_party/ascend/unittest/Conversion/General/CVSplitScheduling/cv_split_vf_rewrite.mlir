// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2 enable-vf-rewrite=false" | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2 enable-vf-rewrite=true" | FileCheck %s --check-prefix=ON

// The optional TD-generated stage folds a shaped idempotent expression before
// graph construction. It does not own scheduling, transfers, or allocation.

// OFF: triton_ascend.cv_split_scheduling.applied = 1 : i32
// OFF: scope.scope
// OFF: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// OFF: scope.scope
// OFF: arith.andi
// OFF: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}

// ON: triton_ascend.cv_split_scheduling.applied = 1 : i32
// ON: scope.scope
// ON: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// ON: scope.scope
// ON-NOT: arith.andi
// ON: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
// ON-NOT: ssbuffer.core_type

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @optional_vf_rewrite(
      %lhs: tensor<32x16xf16>, %rhs: tensor<16x16xf16>,
      %init: tensor<32x16xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %matmul_init = linalg.fill ins(%zero : f32) outs(%init : tensor<32x16xf32>) -> tensor<32x16xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %score = linalg.matmul
          ins(%lhs, %rhs : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%matmul_init : tensor<32x16xf32>) -> tensor<32x16xf32>
      %bits = arith.bitcast %score : tensor<32x16xf32> to tensor<32x16xi32>
      %idempotent = arith.andi %bits, %bits : tensor<32x16xi32>
      %same = arith.bitcast %idempotent : tensor<32x16xi32> to tensor<32x16xf32>
      %vector = math.exp %same : tensor<32x16xf32>
    }
    return
  }
}
