// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s
// RUN: triton-opt %s --debug-only=cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=NO-CREDIT

// C2V FixPipe starts at the first legal point after its Cube producer: a
// loop-wrap ownership wait may appear between them. V2C packing precedes the
// UB-to-L1 copy, whose ready signal is emitted immediately after the move.

// DIAG: [cv-split] built and classified canonical graph
// DIAG: [cv-split] scheduled
// DIAG-DAG: resource=FixPipe op=virtual-transfer
// DIAG-DAG: resource=Vector op=virtual-pack
// DIAG-DAG: resource=UBMove op=virtual-transfer

// CHECK: triton_ascend.cv_split_scheduling.applied = 1 : i32
// CHECK: triton_ascend.cv_split_scheduling.preserve_explicit_schedule
// CHECK-LABEL: func.func @asap_transfers
// CHECK: scope.scope
// CHECK: hivm.hir.convert_layout
// CHECK: memref.memory_space_cast
// CHECK: scf.for
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[SCORE:.*]] = linalg.matmul
// CHECK-NEXT: hivm.hir.fixpipe {{.*}} ins(%[[SCORE]]
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[SCORE_READY:[0-9]+]]
// CHECK: memref.copy %[[V_SRC:.*]], %{{.*}} : memref<16x16xf16> to memref<16x16xf16>
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[P_READY:[0-9]+]]
// CHECK: %[[PRODUCT:.*]] = linalg.matmul
// CHECK-NEXT: hivm.hir.fixpipe {{.*}} ins(%[[PRODUCT]]
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[PRODUCT_READY:[0-9]+]]
// CHECK: scope.scope
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[SCORE_READY]]
// CHECK: tensor.reshape
// CHECK: memref.collapse_shape
// CHECK: bufferization.to_tensor {{.*}} restrict writable
// CHECK: linalg.transpose
// CHECK-NOT: bufferization.materialize_in_destination
// CHECK: hivm.hir.copy
// CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = [[P_READY]]
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = [[PRODUCT_READY]]
// CHECK-NOT: ssbuffer.core_type

// A fresh slot for every logical value needs readiness, but no artificial
// reverse credit or same-engine reuse event.
// NO-CREDIT-LABEL: func.func @asap_transfers
// NO-CREDIT-NOT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>]
// NO-CREDIT-NOT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>]
// NO-CREDIT-NOT: hivm.hir.set_flag
// NO-CREDIT-NOT: hivm.hir.wait_flag
// NO-CREDIT: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @asap_transfers(
      %q: tensor<32x16xf16>, %k: tensor<16x16xf16>,
      %v_src: memref<16x16xf16>, %score_init: tensor<32x16xf32>,
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
      %v_buffer = memref.alloc() : memref<16x16xf16>
      memref.copy %v_src, %v_buffer : memref<16x16xf16> to memref<16x16xf16>
      %v = bufferization.to_tensor %v_buffer restrict writable :
          memref<16x16xf16> to tensor<16x16xf16>
      %product = linalg.matmul
          ins(%p16, %v : tensor<32x16xf16>, tensor<16x16xf16>)
          outs(%product_zero : tensor<32x16xf32>) -> tensor<32x16xf32>
      %sink = math.exp %product : tensor<32x16xf32>
    }
    return
  }
}
