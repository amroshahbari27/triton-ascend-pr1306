// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" | FileCheck %s --check-prefix=FALLBACK
//
// This captured BM=32, BN=32, HD=64 graph has asymmetric C-to-V result
// geometries (score width 32, product width 64).  The qualified materializer
// currently accepts only one uniform C-to-V geometry, so it must preserve the
// original SSA loop instead of committing a slower partially qualified plan.

// FALLBACK-NOT: triton_ascend.cv_split_scheduling.applied
// FALLBACK-NOT: scope.scope
// FALLBACK-NOT: hivm.hir.fixpipe
// FALLBACK-LABEL: func.func @_attn_fwd
// FALLBACK: scf.for
// FALLBACK: linalg.matmul

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c0_i32 = arith.constant 0 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    scf.for %arg13 = %arg10 to %c1_i32 step %c32_i32  : i32 {
      %8 = arith.divsi %arg13, %c256_i32 : i32
      %9 = arith.remsi %arg13, %c256_i32 : i32
      %10 = arith.extsi %8 : i32 to i64
      %11 = arith.muli %10, %c524288_i64 : i64
      %12 = arith.index_cast %11 : i64 to index
      %13 = arith.muli %9, %c32_i32 : i32
      %14 = arith.maxsi %13, %c0_i32 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c64 : index
      %17 = arith.addi %16, %12 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x64xf16>
      memref.copy %reinterpret_cast, %alloc : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
      %18 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
      %19:5 = scf.for %arg14 = %c0_i32 to %c8192_i32 step %c32_i32 iter_args(%arg15 = %7, %arg16 = %1, %arg17 = %6, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32)  : i32 {
        %28 = arith.maxsi %arg18, %c0_i32 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = arith.muli %29, %c64 : index
        %31 = arith.addi %30, %12 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %32 = arith.maxsi %arg19, %c0_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = arith.muli %33, %c64 : index
        %35 = arith.addi %34, %12 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [32, 64], strides: [64, 1] : memref<?xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<32x64xf16>
        memref.copy %reinterpret_cast_6, %alloc_7 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %36 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x64xf16> to tensor<32x64xf16>
        %37 = tensor.empty() : tensor<64x32xf16>
        %transposed = linalg.transpose ins(%36 : tensor<32x64xf16>) outs(%37 : tensor<64x32xf16>) permutation = [1, 0]
        %38 = linalg.matmul {input_precision = "ieee"} ins(%18, %transposed : tensor<32x64xf16>, tensor<64x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %39 = arith.mulf %38, %3 : tensor<32x32xf32>
        %reduced = linalg.reduce ins(%39 : tensor<32x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %54 = arith.maximumf %in, %init : f32
            linalg.yield %54 : f32
          }
        %40 = arith.maximumf %arg17, %reduced : tensor<32xf32>
        %broadcasted_8 = linalg.broadcast ins(%40 : tensor<32xf32>) outs(%2 : tensor<32x32xf32>) dimensions = [1]
        %41 = arith.subf %39, %broadcasted_8 : tensor<32x32xf32>
        %42 = math.exp %41 : tensor<32x32xf32>
        %43 = arith.truncf %42 : tensor<32x32xf32> to tensor<32x32xf16>
        %alloc_9 = memref.alloc() : memref<32x64xf16>
        memref.copy %reinterpret_cast_5, %alloc_9 : memref<32x64xf16, strided<[64, 1], offset: ?>> to memref<32x64xf16>
        %44 = bufferization.to_tensor %alloc_9 restrict writable : memref<32x64xf16> to tensor<32x64xf16>
        %45 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_10 = linalg.reduce ins(%42 : tensor<32x32xf32>) outs(%45 : tensor<32xf32>) dimensions = [1]
          (%in: f32, %init: f32) {
            %54 = arith.addf %in, %init : f32
            linalg.yield %54 : f32
          }
        %46 = arith.subf %arg17, %40 : tensor<32xf32>
        %47 = math.exp %46 : tensor<32xf32>
        %48 = arith.mulf %arg15, %47 : tensor<32xf32>
        %49 = arith.addf %48, %reduced_10 : tensor<32xf32>
        %broadcasted_11 = linalg.broadcast ins(%47 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]
        %50 = arith.mulf %arg16, %broadcasted_11 : tensor<32x64xf32>
        %51 = linalg.matmul {input_precision = "ieee"} ins(%43, %44 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%50 : tensor<32x64xf32>) -> tensor<32x64xf32>
        %52 = arith.addi %arg18, %c32_i32 : i32
        %53 = arith.addi %arg19, %c32_i32 : i32
        scf.yield %49, %51, %40, %52, %53 : tensor<32xf32>, tensor<32x64xf32>, tensor<32xf32>, i32, i32
      } {tt.divisibility_arg1 = dense<32> : tensor<1xi32>}
      %broadcasted = linalg.broadcast ins(%19#0 : tensor<32xf32>) outs(%0 : tensor<32x64xf32>) dimensions = [1]
      %20 = arith.divf %19#1, %broadcasted : tensor<32x64xf32>
      %21 = math.log %19#0 : tensor<32xf32>
      %22 = arith.addf %19#2, %21 : tensor<32xf32>
      %23 = arith.muli %8, %c8192_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %13 : i32 to index
      %26 = arith.addi %24, %25 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %22 in writable %reinterpret_cast_4 : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
      %27 = arith.truncf %20 : tensor<32x64xf32> to tensor<32x64xf16>
      bufferization.materialize_in_destination %27 in writable %reinterpret_cast_3 : (tensor<32x64xf16>, memref<32x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}
