// RUN: triton-opt %s --l1-cache-opt | FileCheck %s
//
// RMSNorm two-pass pattern: loop1 loads X for sum-of-squares, loop2 reloads X.
// L1CacheOpt stages GM->L1 via ND2NZ in loop1 and replaces GM->UB with L12UB in loop2.

// CHECK-LABEL: func.func @rms_norm_fwd_l1
// CHECK-SAME: mix_mode = "mix"
// CHECK-SAME: triton.l1_cache_opt

// CHECK: memref.alloc() : memref<1x32x16x16xf16, #hivm.address_space<cbuf>>
// CHECK-NEXT: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<1x32x16x16xf16, #hivm.address_space<cbuf>>

// CHECK: hivm.hir.nd2nz {dst_continuous}
// CHECK-SAME: #hivm.address_space<cbuf>>

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK: hivm.hir.l12ub ins(%{{.*}} : memref<1x16x16x16xf16, {{.*}} #hivm.address_space<cbuf>>) outs(%{{.*}} : memref<256x16xf16, #hivm.address_space<ub>>)
// CHECK: hivm.hir.nd2nz {dst_continuous}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @rms_norm_fwd_l1(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg3: i32, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: f32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %cst_0 = arith.constant dense<256> : tensor<1xi64>
    %c4095_i32 = arith.constant 4095 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = arith.addi %arg9, %c4095_i32 : i32
    %5 = arith.divsi %4, %c4096_i32 : i32
    %6 = arith.muli %arg14, %arg3 : i32
    %7 = arith.muli %arg14, %arg5 : i32
    %8 = arith.muli %arg14, %arg8 : i32
    %9 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf16>
    memref.copy %reinterpret_cast, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %11 = linalg.matmul {input_precision = "ieee"} ins(%10, %10 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %reshape = tensor.reshape %11(%cst_0) : (tensor<16x16xf32>, tensor<1xi64>) -> tensor<256xf32>
    %12 = bufferization.alloc_tensor() : tensor<f32>
    %13 = linalg.fill ins(%cst_2 : f32) outs(%12 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%reshape : tensor<256xf32>) outs(%13 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %24 = arith.addf %in, %init : f32
        linalg.yield %24 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %inserted = tensor.insert %extracted into %0[%c0] : tensor<1xf32>
    %14 = arith.mulf %inserted, %1 : tensor<1xf32>
    %extracted_3 = tensor.extract %14[%c0] : tensor<1xf32>
    %15 = scf.for %arg17 = %c0_i32 to %5 step %c1_i32 iter_args(%arg18 = %extracted_3) -> (f32)  : i32 {
      %24 = arith.muli %arg17, %c4096_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.addi %9, %25 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<4096xf16>
      %27 = arith.addi %25, %c4096 : index
      %28 = arith.index_cast %arg9 : i32 to index
      %29 = arith.maxsi %25, %28 : index
      %30 = arith.minsi %27, %29 : index
      %31 = arith.subi %30, %25 : index
      %32 = arith.cmpi slt, %31, %c4096 : index
      scf.if %32 {
        linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<4096xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_11[0] [%31] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_13 = memref.subview %alloc_12[0] [%31] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_13 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %33 = bufferization.to_tensor %alloc_12 restrict writable : memref<4096xf16>
      %34 = arith.extf %33 : tensor<4096xf16> to tensor<4096xf32>
      %35 = arith.mulf %34, %34 : tensor<4096xf32>
      %36 = bufferization.alloc_tensor() : tensor<f32>
      %37 = linalg.fill ins(%cst_2 : f32) outs(%36 : tensor<f32>) -> tensor<f32>
      %reduced_14 = linalg.reduce ins(%35 : tensor<4096xf32>) outs(%37 : tensor<f32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %39 = arith.addf %in, %init : f32
          linalg.yield %39 : f32
        }
      %extracted_15 = tensor.extract %reduced_14[] : tensor<f32>
      %inserted_16 = tensor.insert %arg18 into %0[%c0] : tensor<1xf32>
      %inserted_17 = tensor.insert %extracted_15 into %0[%c0] : tensor<1xf32>
      %38 = arith.addf %inserted_16, %inserted_17 : tensor<1xf32>
      %extracted_18 = tensor.extract %38[%c0] : tensor<1xf32>
      scf.yield %extracted_18 : f32
    }
    %16 = arith.sitofp %arg9 : i32 to f32
    %17 = arith.divf %15, %16 : f32
    %inserted_4 = tensor.insert %17 into %0[%c0] : tensor<1xf32>
    %inserted_5 = tensor.insert %arg10 into %0[%c0] : tensor<1xf32>
    %18 = arith.addf %inserted_4, %inserted_5 : tensor<1xf32>
    %extracted_6 = tensor.extract %18[%c0] : tensor<1xf32>
    %inserted_7 = tensor.insert %extracted_6 into %0[%c0] : tensor<1xf32>
    %19 = math.sqrt %inserted_7 : tensor<1xf32>
    %extracted_8 = tensor.extract %19[%c0] : tensor<1xf32>
    %20 = arith.divf %cst_1, %extracted_8 : f32
    %inserted_9 = tensor.insert %20 into %0[%c0] : tensor<1xf32>
    %21 = arith.index_cast %8 : i32 to index
    %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %inserted_9 in writable %reinterpret_cast_10 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
    %22 = tensor.empty() : tensor<4096xf32>
    %23 = linalg.fill ins(%20 : f32) outs(%22 : tensor<4096xf32>) -> tensor<4096xf32>
    scf.for %arg17 = %c0_i32 to %5 step %c1_i32  : i32 {
      %24 = arith.muli %arg17, %c4096_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.addi %9, %25 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_12 = memref.alloc() : memref<4096xf16>
      %27 = arith.addi %25, %c4096 : index
      %28 = arith.index_cast %arg9 : i32 to index
      %29 = arith.maxsi %25, %28 : index
      %30 = arith.minsi %27, %29 : index
      %31 = arith.subi %30, %25 : index
      %32 = arith.cmpi slt, %31, %c4096 : index
      scf.if %32 {
        linalg.fill ins(%cst : f16) outs(%alloc_12 : memref<4096xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast_11[0] [%31] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_13 = memref.subview %alloc_12[0] [%31] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview, %subview_13 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %33 = bufferization.to_tensor %alloc_12 restrict writable : memref<4096xf16>
      %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %alloc_15 = memref.alloc() : memref<4096xf16>
      scf.if %32 {
        linalg.fill ins(%cst : f16) outs(%alloc_15 : memref<4096xf16>)
      } {hivm.unlikely_condition}
      %subview_16 = memref.subview %reinterpret_cast_14[0] [%31] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_17 = memref.subview %alloc_15[0] [%31] [1] : memref<4096xf16> to memref<?xf16, strided<[1]>>
      memref.copy %subview_16, %subview_17 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
      %34 = bufferization.to_tensor %alloc_15 restrict writable : memref<4096xf16>
      %35 = arith.extf %33 : tensor<4096xf16> to tensor<4096xf32>
      %36 = arith.mulf %35, %23 : tensor<4096xf32>
      %37 = arith.extf %34 : tensor<4096xf16> to tensor<4096xf32>
      %38 = arith.mulf %36, %37 : tensor<4096xf32>
      %39 = arith.index_cast %6 : i32 to index
      %40 = arith.addi %39, %25 : index
      %reinterpret_cast_18 = memref.reinterpret_cast %arg2 to offset: [%40], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %41 = arith.truncf %38 : tensor<4096xf32> to tensor<4096xf16>
      %extracted_slice = tensor.extract_slice %41[0] [%31] [1] : tensor<4096xf16> to tensor<?xf16>
      %subview_19 = memref.subview %reinterpret_cast_18[0] [%31] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_19 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

