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
// Generated function: test_elementwise_add
// ====================================================================

void *test_elementwise_add(void *input_arg0, void *input_arg1) {
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

  // Tensor arg1: shape (batch=2048, M=128, K=1)
  int min_stride_arg1 = min_stride1(1, WIDTH_8);
  Tensor tensor_arg1 = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_arg1,
    .byte_stride2 = min_stride_arg1 * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc: shape (batch=2048, M=128, K=1)
  int min_stride_alloc = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_alloc,
    .byte_stride2 = min_stride_alloc * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Set memory addresses for input tensors
  uint32_t start_addr = 0xa0000000;  // PSRAM

  tensor_arg0.base_addr = start_addr;
  uint32_t tensor_size_arg0 = getTensorSize(&tensor_arg0);
  tensor_arg1.base_addr = start_addr + tensor_size_arg0;
  uint32_t tensor_size_arg1 = getTensorSize(&tensor_arg1);

  // Set memory addresses for output tensors

  tensor_alloc.base_addr = start_addr + tensor_size_arg0 + tensor_size_arg1;
  uint32_t tensor_size_alloc = getTensorSize(&tensor_alloc);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: tensor_tensor_operator
  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_arg0, &tensor_arg1, &tensor_alloc, OPERATION_ADD);

  // Return output tensor
  return (void *)tensor_alloc.base_addr;
}
