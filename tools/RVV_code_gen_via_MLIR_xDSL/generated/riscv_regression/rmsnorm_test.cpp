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
// Generated function: rmsnorm_test
// ====================================================================

void *rmsnorm_test(void *input_arg0) {
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

  // Tensor alloc_square: shape (batch=2048, M=128, K=1)
  int min_stride_alloc_square = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_square = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_alloc_square,
    .byte_stride2 = min_stride_alloc_square * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_sum: shape (batch=1, M=128, K=1)
  int min_stride_alloc_sum = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_sum = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 1,
    .byte_stride1 = min_stride_alloc_sum,
    .byte_stride2 = min_stride_alloc_sum * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_mean: shape (batch=1, M=128, K=1)
  int min_stride_alloc_mean = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_mean = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 1,
    .byte_stride1 = min_stride_alloc_mean,
    .byte_stride2 = min_stride_alloc_mean * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_eps: shape (batch=1, M=128, K=1)
  int min_stride_alloc_eps = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_eps = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 1,
    .byte_stride1 = min_stride_alloc_eps,
    .byte_stride2 = min_stride_alloc_eps * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_rsqrt: shape (batch=1, M=128, K=1)
  int min_stride_alloc_rsqrt = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_rsqrt = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 1,
    .byte_stride1 = min_stride_alloc_rsqrt,
    .byte_stride2 = min_stride_alloc_rsqrt * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_normalized: shape (batch=2048, M=128, K=1)
  int min_stride_alloc_normalized = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_normalized = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_alloc_normalized,
    .byte_stride2 = min_stride_alloc_normalized * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Tensor alloc_output: shape (batch=2048, M=128, K=1)
  int min_stride_alloc_output = min_stride1(1, WIDTH_8);
  Tensor tensor_alloc_output = (Tensor){
    .base_addr = -1,
    .dim0      = 1,
    .dim1      = 128,
    .dim2      = 2048,
    .byte_stride1 = min_stride_alloc_output,
    .byte_stride2 = min_stride_alloc_output * 128,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_FP
  };

  // Set memory addresses for reduce operation (using fixed scratchpad)
  // Input tensor: scratchpad0 (0x90000000)
  // Output tensor: scratchpad1 (0x90020000)

  tensor_arg0.base_addr = 0x90000000;  // scratchpad0
  tensor_alloc_square.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_sum.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_mean.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_eps.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_rsqrt.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_normalized.base_addr = 0x90020000;  // scratchpad1
  tensor_alloc_output.base_addr = 0x90020000;  // scratchpad1

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: square_operator
  square_operator(&tensor_arg0, &tensor_alloc_square);

  // Call NPU operator: reduce_dim2_sum
  reduce_dim2_sum(&tensor_alloc_square, &tensor_alloc_sum);

  // Call NPU operator: tensor_imm_operator
  tensor_imm_operator(&tensor_alloc_sum, &tensor_alloc_mean, 2.048000e+03, WIDTH_8, TYPE_FP, OPERATION_DIV);

  // Call NPU operator: lut_squareroot
  lut_squareroot(&tensor_alloc_eps, &tensor_alloc_rsqrt);

  // Call NPU operator: tensor_tensor_operator
  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_arg0, &tensor_alloc_rsqrt, &tensor_alloc_normalized, OPERATION_MUL);

  // Call NPU operator: tensor_tensor_operator
  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_gamma, &tensor_alloc_normalized, &tensor_alloc_output, OPERATION_MUL);

  // Return output tensor
  return (void *)tensor_alloc_output.base_addr;
}
