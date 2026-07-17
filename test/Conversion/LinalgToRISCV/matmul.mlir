//RUN: torch-mlir-opt <%s -convert-linalg-to-rair -rair-verify-lifetimes | FileCheck %s
// CHECK: func.func @forward
// CHECK: %[[CTX:.*]] = rair.acquire {accelerator = "default"} : !rair.context
// CHECK: %[[ALLOC:.*]] = rair.alloc : memref<16x64xf32>
// CHECK: linalg.fill
// CHECK: %[[LHS_LOCAL:.*]] = rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>} : memref<16x32xf32, #rair.space<lmem>>
// CHECK: %[[RHS_LOCAL:.*]] = rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>} : memref<32x64xf32, #rair.space<lmem>>
// CHECK: %[[OUT_LOCAL:.*]] = rair.alloc_buffer %[[CTX]] {memory_space = #rair.space<lmem>} : memref<16x64xf32, #rair.space<lmem>>
// CHECK: rair.transfer %arg0 to %[[LHS_LOCAL]] {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>} : memref<16x32xf32, strided<[?, ?], offset: ?>>, memref<16x32xf32, #rair.space<lmem>>
// CHECK: rair.transfer %arg1 to %[[RHS_LOCAL]] {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>} : memref<32x64xf32, strided<[?, ?], offset: ?>>, memref<32x64xf32, #rair.space<lmem>>
// CHECK: rair.transfer %[[ALLOC]] to %[[OUT_LOCAL]] {dst_memory_space = #rair.space<lmem>, src_memory_space = #rair.space<gmem>} : memref<16x64xf32>, memref<16x64xf32, #rair.space<lmem>>
// CHECK: rair.matmul ins(%[[LHS_LOCAL]], %[[RHS_LOCAL]] : memref<16x32xf32, #rair.space<lmem>>, memref<32x64xf32, #rair.space<lmem>>) outs(%[[OUT_LOCAL]] : memref<16x64xf32, #rair.space<lmem>>)
// CHECK: rair.transfer %[[OUT_LOCAL]] to %[[ALLOC]] {dst_memory_space = #rair.space<gmem>, src_memory_space = #rair.space<lmem>} : memref<16x64xf32, #rair.space<lmem>>, memref<16x64xf32>
// CHECK: rair.dealloc_buffer %[[CTX]], %[[LHS_LOCAL]] : memref<16x32xf32, #rair.space<lmem>>
// CHECK: rair.dealloc_buffer %[[CTX]], %[[RHS_LOCAL]] : memref<32x64xf32, #rair.space<lmem>>
// CHECK: rair.dealloc_buffer %[[CTX]], %[[OUT_LOCAL]] : memref<16x64xf32, #rair.space<lmem>>
// CHECK: rair.release %[[CTX]] : !rair.context
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "MatmulOnly"} {
  func.func @forward(%arg0: memref<16x32xf32, strided<[?, ?], offset: ?>>,
                     %arg1: memref<32x64xf32, strided<[?, ?], offset: ?>>) -> memref<16x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<16x64xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<16x32xf32, strided<[?, ?], offset: ?>>,
                            memref<32x64xf32, strided<[?, ?], offset: ?>>)
                 outs(%alloc : memref<16x64xf32>)
    return %alloc : memref<16x64xf32>
  }
}
