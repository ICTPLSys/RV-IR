// Auto-generated C code from Linalg MLIR
// This code calls NPU SDK operators

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <npu_highlevel.h>
#include <primitive.h>

// ====================================================================
// Tensor Helper Types and Functions
// ====================================================================

// Helper function to compute minimum stride
static inline int min_stride1(int dim, int width) {
    uint32_t size_dim0a = 256 >> width;
    uint32_t size_dim0b = (dim + size_dim0a - 1) / size_dim0a;
    return size_dim0b * 32;
}

// ====================================================================
// Generated function: test_reduce_max
// ====================================================================

void *test_reduce_max(void *input_arg0) {
  // Initialize NPU memory
  npu_mem_init();

  // ====================================================================
  // Tensor Declarations
  // ====================================================================

  // Tensor arg0: shape (batch=2048, M=128, K=1)
  int min_stride_arg0 = min_stride1(1, WIDTH_8);
  Tensor tensor_arg0 = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_arg0,
    .byte_stride2 = min_stride_arg0 * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc: shape (batch=1, M=128, K=1)
  int min_stride_alloc = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 1,
    .byte_stride1 = min_stride_alloc,
    .byte_stride2 = min_stride_alloc * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Set memory addresses for reduce operation (using fixed scratchpad)
  // Input tensor: scratchpad0 (0x90000000)
  // Output tensor: scratchpad1 (0x90020000)

  tensor_arg0.base_addr = 0x90000000;  // scratchpad0
  tensor_alloc.base_addr = 0x90020000;  // scratchpad1

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: reduce_dim2_max
  reduce_dim2_max(&tensor_arg0, &tensor_alloc);

  // Return output tensor
  return (void *)tensor_alloc.base_addr;
}
