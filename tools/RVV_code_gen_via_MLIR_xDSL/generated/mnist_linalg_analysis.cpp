// Auto-generated C code from Linalg MLIR
// This code calls NPU SDK operators

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <datatypes.h>
#include <npu_highlevel.h>
#include <primitive.h>

// SPAD / CIM (same as NPU_SDK test_resnet / test_case_i8_conv_convert)
#define BASE_SCRATCHPAD0 0x90000000u
#define BASE_SCRATCHPAD1 0x90020000u
#define BASE_SCRATCHPAD2 0x90040000u
#define BASE_SCRATCHPAD3 0x90060000u
#define BASE_CIM0 0x00080000u
#define CIM_PAGE_BYTES 0x00002000u

// ====================================================================
// Tensor Helper Types and Functions
// ====================================================================

// min_stride1 is provided by primitive.h; make_tensor matches test_resnet.c

static inline void make_tensor(Tensor *t, uint32_t base_addr,
                              int dim0, int dim1, int dim2,
                              int type_data, int wd_data) {
    int min_stride = min_stride1(dim0, wd_data);
    t->base_addr = base_addr;
    t->dim0 = dim0;
    t->dim1 = dim1;
    t->dim2 = dim2;
    t->type_data = type_data;
    t->wd_data = wd_data;
    t->byte_stride1 = min_stride;
    t->byte_stride2 = min_stride * dim1;
}

// CIM weight tiling for conv (parts) + FC gemm weight base
#define CIM_CONV_PARTS 1u
#define BASE_CIM_FC (BASE_CIM0 + CIM_CONV_PARTS * CIM_PAGE_BYTES)

extern int relu_operator(Tensor *tensor_in, Tensor *tensor_out);

// ====================================================================
// Generated function: forward
// ====================================================================

void forward(void) {
  // Initialize NPU memory
  npu_mem_init();

  const int td = TYPE_INT;
  const int wd = WIDTH_8;

  // ====================================================================
  // Tensor buffers (make_tensor fills layout)
  // ====================================================================

  Tensor tensor_0;
  Tensor tensor_12;
  Tensor tensor_4;
  Tensor tensor_8;
  Tensor tensor_alloc;
  Tensor tensor_alloc_1;
  Tensor tensor_alloc_10;
  Tensor tensor_alloc_11;
  Tensor tensor_alloc_12;
  Tensor tensor_alloc_13;
  Tensor tensor_alloc_14;
  Tensor tensor_alloc_15;
  Tensor tensor_alloc_16;
  Tensor tensor_alloc_17;
  Tensor tensor_alloc_18;
  Tensor tensor_alloc_19;
  Tensor tensor_alloc_2;
  Tensor tensor_alloc_20;
  Tensor tensor_alloc_3;
  Tensor tensor_alloc_4;
  Tensor tensor_alloc_5;
  Tensor tensor_alloc_6;
  Tensor tensor_alloc_7;
  Tensor tensor_alloc_8;
  Tensor tensor_alloc_9;
  Tensor tensor_collapse_shape;
  Tensor tensor_expand_shape;

  // Reduce fusion path: fixed scratchpads

  make_tensor(&tensor_0, BASE_SCRATCHPAD1, 32, 1, 1, td, wd);
  make_tensor(&tensor_4, BASE_SCRATCHPAD1, 32, 196, 1, td, wd);
  make_tensor(&tensor_8, BASE_SCRATCHPAD1, 10, 1, 1, td, wd);
  make_tensor(&tensor_12, BASE_SCRATCHPAD1, 10, 32, 1, td, wd);
  make_tensor(&tensor_alloc, BASE_SCRATCHPAD1, 1, 14, 14, td, wd);
  make_tensor(&tensor_alloc_1, BASE_SCRATCHPAD1, 2, 2, 1, td, wd);
  make_tensor(&tensor_alloc_2, BASE_SCRATCHPAD1, 1, 14, 14, td, wd);
  make_tensor(&tensor_alloc_3, BASE_SCRATCHPAD1, 196, 32, 1, td, wd);
  make_tensor(&tensor_alloc_4, BASE_SCRATCHPAD1, 1, 32, 1, td, wd);
  make_tensor(&tensor_alloc_5, BASE_SCRATCHPAD1, 1, 32, 1, td, wd);
  make_tensor(&tensor_alloc_6, BASE_SCRATCHPAD1, 1, 32, 1, td, wd);
  make_tensor(&tensor_alloc_7, BASE_SCRATCHPAD1, 1, 32, 1, td, wd);
  make_tensor(&tensor_alloc_8, BASE_SCRATCHPAD1, 32, 10, 1, td, wd);
  make_tensor(&tensor_alloc_9, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_alloc_10, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_alloc_11, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_alloc_12, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_13, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_14, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_15, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_16, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_alloc_17, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_alloc_18, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_19, BASE_SCRATCHPAD1, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_20, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);
  make_tensor(&tensor_collapse_shape, BASE_SCRATCHPAD1, 1, 196, 1, td, wd);
  make_tensor(&tensor_expand_shape, BASE_SCRATCHPAD1, 1, 10, 1, td, wd);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: flatten_view_operator
  flatten_operator(&tensor_alloc_2);  //1x14x14->196x1x1

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_4, &tensor_alloc_3, 0);  //32x196x1->196x32x1

  // Call NPU operator: matmul_operator
  gemm_operator(&tensor_collapse_shape, &tensor_alloc_3, &tensor_alloc_5, &tensor_alloc_5, 0, 0);  //1x196x1 * 196x32x1 -> 1x32x1

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_5, &tensor_0, &tensor_alloc_6, OPERATION_ADD); //1x32x1 + 32x1x1 -> 1x32x1

  // Call NPU operator: relu_operator
  relu_operator(&tensor_alloc_6, &tensor_alloc_7); //1x32x1 -> 1x32x1

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_12, &tensor_alloc_8, 0);  //10x32x1->32x10x1

  // Call NPU operator: matmul_operator
  gemm_operator(&tensor_alloc_7, &tensor_alloc_8, &tensor_alloc_10, &tensor_alloc_10, 0, 0);  //1x32x1 * 32x10x1 -> 1x10x1

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_10, &tensor_8, &tensor_alloc_11, OPERATION_ADD); //1x10x1 + 10x1x1 -> 1x10x1

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_11, &tensor_expand_shape, &tensor_alloc_16, OPERATION_SUB); //1x10x1 - 1x10x1 -> 1x10x1

  // Call NPU operator: lut_exp
  lut_exp(&tensor_alloc_16, &tensor_alloc_17); //1x10x1 -> 1x10x1

  // Call NPU operator: reduce_dim1_sum
  reduce_dim1_sum(&tensor_alloc_17, &tensor_alloc_19); //1x10x1 -> 1x1x1

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_17, &tensor_alloc_19, &tensor_alloc_20, OPERATION_DIV);

}
