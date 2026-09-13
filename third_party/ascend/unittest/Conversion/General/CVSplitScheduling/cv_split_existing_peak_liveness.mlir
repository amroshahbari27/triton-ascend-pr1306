// RUN: triton-opt %s --debug-only=cv-split-ssa-buffer-allocation "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG
// RUN: triton-opt %s "--cv_split_scheduling=compile-on-910-95=true unroll-factor=2" | FileCheck %s

// Unrolling creates two 32 KiB materializations for each of two load phases.
// The phases do not overlap, so the allocator charges a 64 KiB scheduled peak
// instead of the 128 KiB sum of all four allocations.  All transfer slots then
// fit without relying on a kernel-specific storage rule.

// DIAG: [cv-split-buffer] existing UB bytes: permanent=0 scheduled-peak=65536
// DIAG: [cv-split-buffer] owned capacity: L0C=65536/262144 UB=106496/253952 L1=16384/524288; downstream PlanMemory required
// DIAG: [cv-split-buffer] assigned 6 block events
// DIAG: [cv-split-buffer] lowered 6 transfers into 12 unified slots

// CHECK: triton_ascend.cv_split_scheduling.applied = 1 : i32
// CHECK-LABEL: func.func @scheduled_existing_peak
// CHECK-COUNT-4: memref.alloc() {{.*}}memref<128x128xf16>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @scheduled_existing_peak(
      %q: tensor<32x128xf16>, %k_source: memref<128x128xf16>,
      %v_source: memref<128x128xf16>, %score_init: tensor<32x128xf32>,
      %product_init: tensor<32x128xf32>) attributes {mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %score_zero = linalg.fill ins(%zero : f32) outs(%score_init : tensor<32x128xf32>) -> tensor<32x128xf32>
    %product_zero = linalg.fill ins(%zero : f32) outs(%product_init : tensor<32x128xf32>) -> tensor<32x128xf32>
    scf.for %iv = %c0 to %c8 step %c1 {
      %k_buffer = memref.alloc() : memref<128x128xf16>
      memref.copy %k_source, %k_buffer : memref<128x128xf16> to memref<128x128xf16>
      %k = bufferization.to_tensor %k_buffer restrict writable : memref<128x128xf16> to tensor<128x128xf16>
      %score = linalg.matmul
          ins(%q, %k : tensor<32x128xf16>, tensor<128x128xf16>)
          outs(%score_zero : tensor<32x128xf32>) -> tensor<32x128xf32>
      %probability = math.exp %score : tensor<32x128xf32>
      %p16 = arith.truncf %probability : tensor<32x128xf32> to tensor<32x128xf16>
      %v_buffer = memref.alloc() : memref<128x128xf16>
      memref.copy %v_source, %v_buffer : memref<128x128xf16> to memref<128x128xf16>
      %v = bufferization.to_tensor %v_buffer restrict writable : memref<128x128xf16> to tensor<128x128xf16>
      %product = linalg.matmul
          ins(%p16, %v : tensor<32x128xf16>, tensor<128x128xf16>)
          outs(%product_zero : tensor<32x128xf32>) -> tensor<32x128xf32>
      %sink = math.exp %product : tensor<32x128xf32>
    }
    return
  }
}
