//RUN: torch-mlir-opt <%s -convert-linalg-to-rair | FileCheck %s
// CHECK: %[[CTX:.*]] = rair.acquire {accelerator = "default"} : !rair.context
// CHECK: rair.alloc
// CHECK: rair.transpose
// CHECK: rair.alloc
// CHECK: rair.alloc
// CHECK: rair.transfer
// CHECK: rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>}
// CHECK: rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>}
// CHECK: rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>}
// CHECK: rair.transfer {{.*}} {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>}
// CHECK: rair.transfer {{.*}} {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>}
// CHECK: rair.transfer {{.*}} {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>}
// CHECK: rair.matmul
// CHECK: rair.transfer {{.*}} {dst_memory_space = #rair.space<gmem>, src_memory_space = #rair.space<lmem>}
// CHECK: rair.dealloc_buffer
// CHECK: rair.dealloc_buffer
// CHECK: rair.dealloc_buffer
// CHECK: rair.release %[[CTX]]
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "Linear"} {
  memref.global "private" constant @__constant_64x32xf32 : memref<64x32xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<16x32xf32, strided<[?, ?], offset: ?>>) -> memref<16x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_64x32xf32 : memref<64x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
    linalg.transpose ins(%0 : memref<64x32xf32>) outs(%alloc : memref<32x64xf32>) permutation = [1, 0]
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<16x64xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    memref.copy %alloc_1, %alloc_2 : memref<16x64xf32> to memref<16x64xf32>
    linalg.matmul ins(%arg0, %alloc : memref<16x32xf32, strided<[?, ?], offset: ?>>, memref<32x64xf32>) outs(%alloc_2 : memref<16x64xf32>)
    return %alloc_2 : memref<16x64xf32>
  }
}
