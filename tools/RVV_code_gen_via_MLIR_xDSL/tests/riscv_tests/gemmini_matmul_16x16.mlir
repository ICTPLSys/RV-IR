// Gemmini matmul test: 16x16 (single tile, fits in DIM=16)
// Usage: python gemmini_mlir_to_c.py tests/riscv_tests/gemmini_matmul_16x16.mlir -v
module attributes {torch.debug_module_name = "MatmulGemmini16"} {
  func.func @gemmini_matmul_16x16(%arg0: memref<16x16xf32>,
                                    %arg1: memref<16x16xf32>) -> memref<16x16xf32> {
    %cst = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
    rair.matmul ins(%arg0, %arg1 : memref<16x16xf32>, memref<16x16xf32>) outs(%alloc : memref<16x16xf32>)
    return %alloc : memref<16x16xf32>
  }
}
