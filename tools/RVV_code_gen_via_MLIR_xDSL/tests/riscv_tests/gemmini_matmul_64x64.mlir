// Gemmini matmul test: 64x64 (requires tiling with DIM=16)
// Usage: python gemmini_mlir_to_c.py tests/riscv_tests/gemmini_matmul_64x64.mlir -v
module attributes {torch.debug_module_name = "MatmulGemmini64"} {
  func.func @gemmini_matmul_64x64(%arg0: memref<64x32xf32>,
                                    %arg1: memref<32x64xf32>) -> memref<64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
    rair.matmul ins(%arg0, %arg1 : memref<64x32xf32>, memref<32x64xf32>) outs(%alloc : memref<64x64xf32>)
    return %alloc : memref<64x64xf32>
  }
}
