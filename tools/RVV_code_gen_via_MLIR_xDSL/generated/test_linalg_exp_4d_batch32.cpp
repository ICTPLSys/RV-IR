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
// Generated function: test_exp_4d_batch32
// ====================================================================

void *test_exp_4d_batch32(void *input_arg0) {
  // Initialize NPU memory
  npu_mem_init();

  // ====================================================================
  // Tensor Declarations
  // ====================================================================

  // Tensor arg0: 4D shape [batch=32, C=32, H=128, W=128]   
  // Will be processed as 32 separate 3D tensors [C=32, H=128, W=128]
  int min_stride_arg0_3d = min_stride1(32, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_arg0 = 0;

  // Tensor alloc: 4D shape [batch=32, C=32, H=128, W=128] 
  // Will be processed as 32 separate 3D tensors [C=32, H=128, W=128]
  int min_stride_alloc_3d = min_stride1(32, WIDTH_8);
  // 4D tensor base address (will be adjusted in loop)
  uint32_t base_addr_alloc = 0;

  // Set memory addresses for input tensors
  uint32_t start_addr = 0xa0000000;  // PSRAM

  base_addr_arg0 = start_addr;
  uint32_t tensor_size_arg0 = (32 * 32 * 128 * 128 * 4);

  // Set memory addresses for output tensors

  base_addr_alloc = start_addr + tensor_size_arg0;
  uint32_t tensor_size_alloc = (32 * 32 * 128 * 128 * 4);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: lut_exp
  // Processing 4D tensors: looping over batch dimension (size=32)
  for (int batch = 0; batch < 32; batch++) {
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
    lut_exp(&tensor_arg0_3d, &tensor_alloc_3d);
  }

  // Return output tensor
  return (void *)base_addr_alloc;
}
