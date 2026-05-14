// Auto-generated C code from EmitC MLIR
// Using strategy: simple
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <npu_highlevel.h>
#include <primitive.h>

// ====================================================================
// Tensor Helper Types and Functions
// ====================================================================


// Helper functions

// Helper function to compute minimum stride

// ====================================================================
// Generated function: transpose_test
// Strategy: simple
// ====================================================================

void *transpose_test(void *arg0, void *arg1) {
  // Initialize NPU memory
  npu_mem_init();

  // Transpose operation detected

  // Input tensor: shape (1, 128, 8, ...)
  Tensor tensor_transpose_in = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 8,
    .byte_stride1 = 0,
    .byte_stride2 = 0,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Output tensor: shape (1, 8, 128, ...)
  Tensor tensor_transpose_out = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 8,
    .dim2      = 128,
    .byte_stride1 = 0,
    .byte_stride2 = 0,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Set memory addresses for transpose tensors
  tensor_transpose_in.base_addr = 0x90000000;  // scratchpad0
  tensor_transpose_out.base_addr = 0x90020000;  // scratchpad1

  // Perform transpose operation with dim_axis=1
  transpose_operator(&tensor_transpose_in, &tensor_transpose_out, 1);

  // Return pointer to output tensor
  return (void *)tensor_transpose_out.base_addr;
}

// ====================================================================
// Main Function (for testing)
// ====================================================================

int main(void) {
  transpose_test(NULL, NULL);
    return 0;
}
