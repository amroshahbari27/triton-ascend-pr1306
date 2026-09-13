// Minimal static Cube-to-Vector graph used to exercise unroll choices.
// The leading result dimension proves the two-AIV row partition.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @simple_c2v(%lhs: tensor<32x16xf16>,
                        %rhs: tensor<16x16xf16>,
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
      %vector = math.exp %score : tensor<32x16xf32>
    }
    return
  }
}
