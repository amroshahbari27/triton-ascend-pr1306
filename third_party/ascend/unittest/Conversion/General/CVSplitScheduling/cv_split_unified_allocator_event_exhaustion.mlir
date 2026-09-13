// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>/dev/null | FileCheck %s
// RUN: triton-opt %s --debug-only=cv-split-ssa-buffer-allocation,cv-split-scheduling "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// One source iteration has sixteen different C2V result geometries.  The
// qualified lowering supports one uniform C2V geometry per pipeline, so the
// graph admission check rejects the candidate transactionally.
// DIAG: [cv-split] rejecting unqualified asymmetric Cube-to-Vector geometry
// DIAG: [cv-split-scheduling]: BuildSSAGraph rejected event_exhaustion_rolls_back

// CHECK-NOT: triton_ascend.cv_split_scheduling.applied
// CHECK-LABEL: func.func @event_exhaustion_rolls_back
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[ONE]] {
// CHECK-COUNT-16: linalg.matmul
// CHECK: }
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.sync_block

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @event_exhaustion_rolls_back(
      %lhs: tensor<2x1xf16>,
      %r0: tensor<1x16xf16>, %i0: tensor<2x16xf32>,
      %r1: tensor<1x32xf16>, %i1: tensor<2x32xf32>,
      %r2: tensor<1x48xf16>, %i2: tensor<2x48xf32>,
      %r3: tensor<1x64xf16>, %i3: tensor<2x64xf32>,
      %r4: tensor<1x80xf16>, %i4: tensor<2x80xf32>,
      %r5: tensor<1x96xf16>, %i5: tensor<2x96xf32>,
      %r6: tensor<1x112xf16>, %i6: tensor<2x112xf32>,
      %r7: tensor<1x128xf16>, %i7: tensor<2x128xf32>,
      %r8: tensor<1x144xf16>, %i8: tensor<2x144xf32>,
      %r9: tensor<1x160xf16>, %i9: tensor<2x160xf32>,
      %ra: tensor<1x176xf16>, %ia: tensor<2x176xf32>,
      %rb: tensor<1x192xf16>, %ib: tensor<2x192xf32>,
      %rc: tensor<1x208xf16>, %ic: tensor<2x208xf32>,
      %rd: tensor<1x224xf16>, %id: tensor<2x224xf32>,
      %re: tensor<1x240xf16>, %ie: tensor<2x240xf32>,
      %rf: tensor<1x256xf16>, %if: tensor<2x256xf32>)
      attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    %z0 = linalg.fill ins(%zero : f32) outs(%i0 : tensor<2x16xf32>) -> tensor<2x16xf32>
    %z1 = linalg.fill ins(%zero : f32) outs(%i1 : tensor<2x32xf32>) -> tensor<2x32xf32>
    %z2 = linalg.fill ins(%zero : f32) outs(%i2 : tensor<2x48xf32>) -> tensor<2x48xf32>
    %z3 = linalg.fill ins(%zero : f32) outs(%i3 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %z4 = linalg.fill ins(%zero : f32) outs(%i4 : tensor<2x80xf32>) -> tensor<2x80xf32>
    %z5 = linalg.fill ins(%zero : f32) outs(%i5 : tensor<2x96xf32>) -> tensor<2x96xf32>
    %z6 = linalg.fill ins(%zero : f32) outs(%i6 : tensor<2x112xf32>) -> tensor<2x112xf32>
    %z7 = linalg.fill ins(%zero : f32) outs(%i7 : tensor<2x128xf32>) -> tensor<2x128xf32>
    %z8 = linalg.fill ins(%zero : f32) outs(%i8 : tensor<2x144xf32>) -> tensor<2x144xf32>
    %z9 = linalg.fill ins(%zero : f32) outs(%i9 : tensor<2x160xf32>) -> tensor<2x160xf32>
    %za = linalg.fill ins(%zero : f32) outs(%ia : tensor<2x176xf32>) -> tensor<2x176xf32>
    %zb = linalg.fill ins(%zero : f32) outs(%ib : tensor<2x192xf32>) -> tensor<2x192xf32>
    %zc = linalg.fill ins(%zero : f32) outs(%ic : tensor<2x208xf32>) -> tensor<2x208xf32>
    %zd = linalg.fill ins(%zero : f32) outs(%id : tensor<2x224xf32>) -> tensor<2x224xf32>
    %ze = linalg.fill ins(%zero : f32) outs(%ie : tensor<2x240xf32>) -> tensor<2x240xf32>
    %zf = linalg.fill ins(%zero : f32) outs(%if : tensor<2x256xf32>) -> tensor<2x256xf32>
    scf.for %iv = %c0 to %c4 step %c1 {
      %v0 = linalg.matmul ins(%lhs, %r0 : tensor<2x1xf16>, tensor<1x16xf16>) outs(%z0 : tensor<2x16xf32>) -> tensor<2x16xf32>
      %s0 = math.exp %v0 : tensor<2x16xf32>
      %v1 = linalg.matmul ins(%lhs, %r1 : tensor<2x1xf16>, tensor<1x32xf16>) outs(%z1 : tensor<2x32xf32>) -> tensor<2x32xf32>
      %s1 = math.exp %v1 : tensor<2x32xf32>
      %v2 = linalg.matmul ins(%lhs, %r2 : tensor<2x1xf16>, tensor<1x48xf16>) outs(%z2 : tensor<2x48xf32>) -> tensor<2x48xf32>
      %s2 = math.exp %v2 : tensor<2x48xf32>
      %v3 = linalg.matmul ins(%lhs, %r3 : tensor<2x1xf16>, tensor<1x64xf16>) outs(%z3 : tensor<2x64xf32>) -> tensor<2x64xf32>
      %s3 = math.exp %v3 : tensor<2x64xf32>
      %v4 = linalg.matmul ins(%lhs, %r4 : tensor<2x1xf16>, tensor<1x80xf16>) outs(%z4 : tensor<2x80xf32>) -> tensor<2x80xf32>
      %s4 = math.exp %v4 : tensor<2x80xf32>
      %v5 = linalg.matmul ins(%lhs, %r5 : tensor<2x1xf16>, tensor<1x96xf16>) outs(%z5 : tensor<2x96xf32>) -> tensor<2x96xf32>
      %s5 = math.exp %v5 : tensor<2x96xf32>
      %v6 = linalg.matmul ins(%lhs, %r6 : tensor<2x1xf16>, tensor<1x112xf16>) outs(%z6 : tensor<2x112xf32>) -> tensor<2x112xf32>
      %s6 = math.exp %v6 : tensor<2x112xf32>
      %v7 = linalg.matmul ins(%lhs, %r7 : tensor<2x1xf16>, tensor<1x128xf16>) outs(%z7 : tensor<2x128xf32>) -> tensor<2x128xf32>
      %s7 = math.exp %v7 : tensor<2x128xf32>
      %v8 = linalg.matmul ins(%lhs, %r8 : tensor<2x1xf16>, tensor<1x144xf16>) outs(%z8 : tensor<2x144xf32>) -> tensor<2x144xf32>
      %s8 = math.exp %v8 : tensor<2x144xf32>
      %v9 = linalg.matmul ins(%lhs, %r9 : tensor<2x1xf16>, tensor<1x160xf16>) outs(%z9 : tensor<2x160xf32>) -> tensor<2x160xf32>
      %s9 = math.exp %v9 : tensor<2x160xf32>
      %va = linalg.matmul ins(%lhs, %ra : tensor<2x1xf16>, tensor<1x176xf16>) outs(%za : tensor<2x176xf32>) -> tensor<2x176xf32>
      %sa = math.exp %va : tensor<2x176xf32>
      %vb = linalg.matmul ins(%lhs, %rb : tensor<2x1xf16>, tensor<1x192xf16>) outs(%zb : tensor<2x192xf32>) -> tensor<2x192xf32>
      %sb = math.exp %vb : tensor<2x192xf32>
      %vc = linalg.matmul ins(%lhs, %rc : tensor<2x1xf16>, tensor<1x208xf16>) outs(%zc : tensor<2x208xf32>) -> tensor<2x208xf32>
      %sc = math.exp %vc : tensor<2x208xf32>
      %vd = linalg.matmul ins(%lhs, %rd : tensor<2x1xf16>, tensor<1x224xf16>) outs(%zd : tensor<2x224xf32>) -> tensor<2x224xf32>
      %sd = math.exp %vd : tensor<2x224xf32>
      %ve = linalg.matmul ins(%lhs, %re : tensor<2x1xf16>, tensor<1x240xf16>) outs(%ze : tensor<2x240xf32>) -> tensor<2x240xf32>
      %se = math.exp %ve : tensor<2x240xf32>
      %vf = linalg.matmul ins(%lhs, %rf : tensor<2x1xf16>, tensor<1x256xf16>) outs(%zf : tensor<2x256xf32>) -> tensor<2x256xf32>
      %sf = math.exp %vf : tensor<2x256xf32>
    }
    return
  }
}
