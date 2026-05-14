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
// Generated function: test_identity_broadcast
// ====================================================================

void *test_identity_broadcast(void *input_arg0) {
  // Initialize NPU memory
  npu_mem_init();

  // ====================================================================
  // Tensor Declarations
  // ====================================================================

  // Tensor arg0: shape (batch=1, M=8192, K=2048)
  int min_stride_arg0 = min_stride1(2048, WIDTH_8);
  Tensor tensor_arg0 = (Tensor){
    .base_addr = -1,
    .dim0      = 2048,
    .dim1      = 8192,
    .dim2      = 1,
    .byte_stride1 = min_stride_arg0,
    .byte_stride2 = min_stride_arg0 * 8192,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc: shape (batch=8192, M=2048, K=1)
  int min_stride_alloc = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 2048,
    .dim2      = 8192,
    .byte_stride1 = min_stride_alloc,
    .byte_stride2 = min_stride_alloc * 2048,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Set memory addresses for input tensors
  uint32_t start_addr = 0xa0000000;  // PSRAM

  tensor_arg0.base_addr = start_addr;
  uint32_t tensor_size_arg0 = getTensorSize(&tensor_arg0);

  // Set memory addresses for output tensors

  tensor_alloc.base_addr = start_addr + tensor_size_arg0;
  uint32_t tensor_size_alloc = getTensorSize(&tensor_alloc);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: broadcast_operator
  // Broadcast/copy tensor data
  // Note: Broadcasting is handled by memcpy with proper memory layout
  memcpy(tensor_alloc.base_addr, tensor_arg0.base_addr, getTensorSize(&tensor_arg0));

  // Return output tensor
  return (void *)tensor_alloc.base_addr;
}
