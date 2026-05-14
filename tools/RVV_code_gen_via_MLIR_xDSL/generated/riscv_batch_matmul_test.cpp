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
// Generated function: forward
// Strategy: simple
// ====================================================================

void *forward(void *arg0, void *arg1) {
  // Initialize NPU memory
  npu_mem_init();

  // Simple GEMM strategy: direct computation

  // Tensor A: shape (batch=1, M=8, K=4)
  int min_stride_A = min_stride1(4, WIDTH_8);
  Tensor tensor_A = (Tensor){
    .base_addr = -1,
    .dim0      = 4,
    .dim1      = 8,
    .dim2      = 1,
    .byte_stride1 = min_stride_A,
    .byte_stride2 = min_stride_A * 8,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Tensor B: shape (batch=1, M=16, K=8)
  int min_stride_B = min_stride1(8, WIDTH_8);
  Tensor tensor_B = (Tensor){
    .base_addr = -1,
    .dim0      = 8,
    .dim1      = 16,
    .dim2      = 1,
    .byte_stride1 = min_stride_B,
    .byte_stride2 = min_stride_B * 16,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Tensor C: shape (batch=1, M=16, K=4)
  int min_stride_C = min_stride1(4, WIDTH_8);
  Tensor tensor_C = (Tensor){
    .base_addr = -1,
    .dim0      = 4,
    .dim1      = 16,
    .dim2      = 1,
    .byte_stride1 = min_stride_C,
    .byte_stride2 = min_stride_C * 16,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Set memory addresses for tensors
   uint32_t start_addr = 0xa0000000;   // psram
  tensor_A.base_addr = 0x90000000;  // scratchpad0
  tensor_B.base_addr = 0x80000;  // CIM
  tensor_C.base_addr = 0x90020000;  // scratchpad1

  // Perform GEMM operation: C = A @ B
  // accumulate=0, activate=0
  gemm_operator(&tensor_A, &tensor_B, &tensor_C, &tensor_C, 0, 0);

  // Return pointer to output tensor C
  return (void *)tensor_C.base_addr;
}

// ====================================================================
// Main Function (for testing)
// ====================================================================

int main(void) {
  forward(NULL, NULL);
    return 0;
}
