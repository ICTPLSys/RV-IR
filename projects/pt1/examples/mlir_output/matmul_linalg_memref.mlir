module attributes {torch.debug_module_name = "MatmulModule"} {
  func.func @forward(%arg0: memref<16x32xf32, strided<[?, ?], offset: ?>>, %arg1: memref<32x64xf32, strided<[?, ?], offset: ?>>) -> memref<16x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<16x64xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    memref.copy %alloc, %alloc_0 : memref<16x64xf32> to memref<16x64xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<16x32xf32, strided<[?, ?], offset: ?>>, memref<32x64xf32, strided<[?, ?], offset: ?>>) outs(%alloc_0 : memref<16x64xf32>)
    return %alloc_0 : memref<16x64xf32>
  }
}
