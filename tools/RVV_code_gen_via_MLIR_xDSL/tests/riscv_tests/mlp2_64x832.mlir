// MLP2 (64x832): Two-layer MLP matching Gemmini's mlp2_64x832_demo
// Layer 0: [64 x 832] * [832 x 832] -> [64 x 832]  (with ReLU)
// Layer 1: [64 x 832] * [832 x 64]  -> [64 x 64]
//
// Usage:
//   python gemmini_mlir_to_c.py tests/riscv_tests/mlp2_64x832.mlir --mode auto -v
module attributes {torch.debug_module_name = "MLP2_64x832"} {
  func.func @mlp2_forward(%input: memref<64x832xf32>,
                           %weights0: memref<832x832xf32>,
                           %weights1: memref<832x64xf32>) -> memref<64x64xf32> {
    %cst = arith.constant 0.0 : f32

    // Layer 0: hidden = input @ weights0   [64x832] * [832x832] -> [64x832]
    %hidden = memref.alloc() : memref<64x832xf32>
    linalg.fill ins(%cst : f32) outs(%hidden : memref<64x832xf32>)
    rair.matmul ins(%input, %weights0 : memref<64x832xf32>, memref<832x832xf32>) outs(%hidden : memref<64x832xf32>)

    // Layer 1: output = hidden @ weights1  [64x832] * [832x64] -> [64x64]
    %output = memref.alloc() : memref<64x64xf32>
    linalg.fill ins(%cst : f32) outs(%output : memref<64x64xf32>)
    rair.matmul ins(%hidden, %weights1 : memref<64x832xf32>, memref<832x64xf32>) outs(%output : memref<64x64xf32>)

    return %output : memref<64x64xf32>
  }
}
