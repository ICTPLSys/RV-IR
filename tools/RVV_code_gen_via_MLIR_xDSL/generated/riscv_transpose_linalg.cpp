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
// Generated function: transpose_test
// ====================================================================

void *transpose_test(void *input_arg0) {
  // Initialize NPU memory
  npu_mem_init();

  // ====================================================================
  // Tensor Declarations
  // ====================================================================

  // Tensor arg0: 4D shape [batch=1, C=128, H=8, W=64]
  // Will be processed as 1 separate 3D tensors [C=128, H=8, W=64]
  int min_stride_arg0_3d = min_stride1(128, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_arg0 = 0;

  // Tensor alloc_input: 4D shape [batch=1, C=128, H=8, W=64]
  // Will be processed as 1 separate 3D tensors [C=128, H=8, W=64]
  int min_stride_alloc_input_3d = min_stride1(128, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_alloc_input = 0;

  // Tensor alloc_output: 4D shape [batch=1, C=8, H=128, W=64]
  // Will be processed as 1 separate 3D tensors [C=8, H=128, W=64]
  int min_stride_alloc_output_3d = min_stride1(8, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_alloc_output = 0;

  // Set memory addresses for input tensors
  uint32_t start_addr = 0xa0000000;  // PSRAM

  base_addr_arg0 = start_addr;
  uint32_t tensor_size_arg0 = (1 * 128 * 8 * 64 * 4);

  // Set memory addresses for output tensors

  base_addr_alloc_input = start_addr + tensor_size_arg0;
  uint32_t tensor_size_alloc_input = (1 * 128 * 8 * 64 * 4);
  base_addr_alloc_output = start_addr + tensor_size_arg0 + tensor_size_alloc_input;
  uint32_t tensor_size_alloc_output = (1 * 8 * 128 * 64 * 4);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: create_tensor_transpose_in
  create_tensor_transpose_in(&tensor_0, &tensor_2, &tensor_3, &tensor_4, &tensor_5);

  // Call NPU operator: create_tensor_transpose_out
  create_tensor_transpose_out(&tensor_1, &tensor_6, &tensor_7, &tensor_8, &tensor_9);

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_10, &tensor_11, &tensor_12);

  // Return output tensor
  return (void *)base_addr_alloc_output;
}
