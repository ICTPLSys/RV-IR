module attributes {torch.debug_module_name = "MatmulModule"} {
  func.func @forward(%arg0: memref<16x32xf32, strided<[?, ?], offset: ?>>, %arg1: memref<32x64xf32, strided<[?, ?], offset: ?>>) -> memref<16x64xf32> {
    %0 = rair.acquire {accelerator = "default"} : !rair.context
    %cst = arith.constant 0.000000e+00 : f32
    %1 = rair.alloc : memref<16x64xf32>
    linalg.fill ins(%cst : f32) outs(%1 : memref<16x64xf32>)
    %2 = rair.alloc : memref<16x64xf32>
    rair.transfer %1 to %2 : memref<16x64xf32>, memref<16x64xf32>
    rair.matmul ins(%arg0, %arg1 : memref<16x32xf32, strided<[?, ?], offset: ?>>, memref<32x64xf32, strided<[?, ?], offset: ?>>) outs(%2 : memref<16x64xf32>)
    rair.release %0 : !rair.context
    return %2 : memref<16x64xf32>
  }
}

