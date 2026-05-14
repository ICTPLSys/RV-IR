// Test file for RISC-V transpose operation
// This tests the transpose operation from 4D tensor

module attributes {torch.debug_module_name = "TransposeTest"} {
  func.func @transpose_test(%arg0: memref<1x128x8x64xf32>) -> memref<1x8x128x64xf32> {
    %cst = arith.constant 0.0 : f32

    %alloc_input = memref.alloc() : memref<1x128x8x64xf32>
    %alloc_output = memref.alloc() : memref<1x8x128x64xf32>

    // Initialize input tensor with some data
    linalg.fill ins(%cst : f32) outs(%alloc_input : memref<1x128x8x64xf32>)

    // Perform transpose: [1, 128, 8, 64] -> [1, 8, 128, 64]
    // Permutation [0, 2, 1, 3] means:
    //   - dim 0 (size 1) stays at position 0
    //   - dim 2 (size 8) moves to position 1
    //   - dim 1 (size 128) moves to position 2
    //   - dim 3 (size 64) stays at position 3
    linalg.transpose ins(%alloc_input : memref<1x128x8x64xf32>) outs(%alloc_output : memref<1x8x128x64xf32>) permutation = [0, 2, 1, 3]
    return %alloc_output : memref<1x8x128x64xf32>
  }
}
