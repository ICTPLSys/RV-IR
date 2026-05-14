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
// Generated function: test_sub_4d_broadcast
// ====================================================================

void *test_sub_4d_broadcast(void *input_arg0, void *input_arg1) {
  // Initialize NPU memory
  npu_mem_init();

  // ====================================================================
  // Tensor Declarations
  // ====================================================================

  // Tensor arg0: 4D shape [batch=1, C=32, H=128, W=128]
  // Will be processed as 1 separate 3D tensors [C=32, H=128, W=128]
  int min_stride_arg0_3d = min_stride1(32, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_arg0 = 0;

  // Tensor arg1: 4D shape [batch=1, C=32, H=128, W=1]
  // Will be processed as 1 separate 3D tensors [C=32, H=128, W=1]
  int min_stride_arg1_3d = min_stride1(32, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_arg1 = 0;

  // Tensor alloc: 4D shape [batch=1, C=32, H=128, W=128]
  // Will be processed as 1 separate 3D tensors [C=32, H=128, W=128]
  int min_stride_alloc_3d = min_stride1(32, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_alloc = 0;

  // Set memory addresses for input tensors
  uint32_t start_addr = 0xa0000000;  // PSRAM

  base_addr_arg0 = start_addr;
  uint32_t tensor_size_arg0 = (1 * 32 * 128 * 128 * 4);
  base_addr_arg1 = start_addr + tensor_size_arg0;
  uint32_t tensor_size_arg1 = (1 * 32 * 128 * 1 * 4);

  // Set memory addresses for output tensors

  base_addr_alloc = start_addr + tensor_size_arg0 + tensor_size_arg1;
  uint32_t tensor_size_alloc = (1 * 32 * 128 * 128 * 4);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: tensor_tensor_sub
  // Processing 4D tensors: looping over batch dimension (size=1)
  for (int batch = 0; batch < 1; batch++) {
    // Create 3D tensor views for current batch
    Tensor tensor_arg0_3d = (Tensor){
      .base_addr = base_addr_arg0 + batch * 32 * 128 * 128 * 4,
      .dim0      = 32,
      .dim1      = 128,
      .dim2      = 128,
      .byte_stride1 = min_stride_arg0_3d,
      .byte_stride2 = min_stride_arg0_3d * 128,
      .wd_data      = WIDTH_8,
      .type_data    = TYPE_FP
    };
    Tensor tensor_arg1_3d = (Tensor){
      .base_addr = base_addr_arg1 + batch * 32 * 128 * 128 * 4,
      .dim0      = 32,
      .dim1      = 128,
      .dim2      = 1,
      .byte_stride1 = min_stride_arg1_3d,
      .byte_stride2 = min_stride_arg1_3d * 128,
      .wd_data      = WIDTH_8,
      .type_data    = TYPE_FP
    };
    Tensor tensor_alloc_3d = (Tensor){
      .base_addr = base_addr_alloc + batch * 32 * 128 * 128 * 4,
      .dim0      = 32,
      .dim1      = 128,
      .dim2      = 128,
      .byte_stride1 = min_stride_alloc_3d,
      .byte_stride2 = min_stride_alloc_3d * 128,
      .wd_data      = WIDTH_8,
      .type_data    = TYPE_FP
    };
    tensor_tensor_sub(&tensor_arg0_3d, &tensor_arg1_3d, &tensor_alloc_3d);
  }

  // Return output tensor
  return (void *)base_addr_alloc;
}
