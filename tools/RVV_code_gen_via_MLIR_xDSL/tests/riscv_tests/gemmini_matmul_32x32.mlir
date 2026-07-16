// Gemmini matmul test: 32x32 (requires tiling with DIM=16)
// Usage: python gemmini_mlir_to_c.py tests/riscv_tests/gemmini_matmul_32x32.mlir -v
module attributes {torch.debug_module_name = "MatmulGemmini32"} {
  func.func @gemmini_matmul_32x32(%arg0: memref<32x32xf32>,
                                    %arg1: memref<32x32xf32>) -> memref<32x32xf32> {
    %cst = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<32x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
    rair.matmul ins(%arg0, %arg1 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc : memref<32x32xf32>)
    return %alloc : memref<32x32xf32>
  }
}
