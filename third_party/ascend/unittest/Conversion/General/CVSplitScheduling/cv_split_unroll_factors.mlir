// RUN: triton-opt %S/Inputs/simple_c2v.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s --check-prefix=U2
// RUN: triton-opt %S/Inputs/simple_c2v.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=U4
// RUN: triton-opt %S/Inputs/simple_c2v.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" | FileCheck %s --check-prefix=U8
// RUN: triton-opt %S/Inputs/simple_c2v.mlir --debug-only=cv-split-ssa-buffer-allocation "--cv_split_scheduling=compile-on-910-95=true unroll-factor=8" 2>&1 >/dev/null | FileCheck %s --check-prefix=U8-EVENT

// Every supported factor commits the same graph/lowering pipeline.  The
// number of Cube and Vector operations is the selected unroll factor.

// U2: triton_ascend.cv_split_scheduling.applied = 1 : i32
// U2: scope.scope
// U2-COUNT-2: linalg.matmul
// U2: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// U2: scope.scope
// U2-COUNT-2: math.exp
// U2: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
// U2-NOT: ssbuffer.core_type

// U4: triton_ascend.cv_split_scheduling.applied = 1 : i32
// U4: scope.scope
// U4-COUNT-4: linalg.matmul
// U4: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// U4: scope.scope
// U4-COUNT-4: math.exp
// U4: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
// U4-NOT: ssbuffer.core_type

// U8: triton_ascend.cv_split_scheduling.applied = 1 : i32
// U8: scope.scope
// U8-COUNT-8: linalg.matmul
// U8: } {hivm.tcore_type = #hivm.tcore_type<CUBE>, noinline}
// U8: scope.scope
// U8-COUNT-8: math.exp
// U8: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
// U8-NOT: ssbuffer.core_type

// With ample capacity, each logical transfer receives a fresh destination and
// one readiness event; no physical-slot reuse credit is needed.
// U8-EVENT: [cv-split-buffer] assigned 8 block events
