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
// SPAD: NUM_BANK*DP_BANK lines of WD_BANK/8 bytes each (Config.py); tensor2buffer requires 32-byte-aligned byte offsets (Golden_model.py).
#define SPAD_BANK_BYTES   0x00020000u
#define SPAD_OFFSET_ALIGN 32u

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
  Tensor tensor_alloc_11__reduce_dim0_in;
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

  // Reduce path: pack activations/scratch in SPAD0..SPAD3 (SPAD_BANK_BYTES per bank, SPAD_OFFSET_ALIGN-byte offsets). CIM weights unchanged.

  // 修正：memref<32xf32> → (32,1,1)
  make_tensor(&tensor_0, BASE_SCRATCHPAD0 + 0x0u, 32, 1, 1, td, wd);
  // 修正：memref<10x32xf32> → (32,10,1)
  make_tensor(&tensor_12, BASE_SCRATCHPAD0 + 0x20u, 32, 10, 1, td, wd);
  // 修正：memref<32x196xf32> → (196,32,1)
  make_tensor(&tensor_4, BASE_SCRATCHPAD0 + 0x420u, 196, 32, 1, td, wd);
  // 修正：memref<10xf32> → (10,1,1)
  make_tensor(&tensor_8, BASE_SCRATCHPAD0 + 0x1CA0u, 10, 1, 1, td, wd);
  // 修正：memref<1x1x14x14xf32> → (1,14,14)
  make_tensor(&tensor_alloc, BASE_SCRATCHPAD0 + 0x1CC0u, 14, 14, 1, td, wd);
  // 修正：memref<2x2xf32> → (2,2,1)
  make_tensor(&tensor_alloc_1, BASE_SCRATCHPAD0 + 0x3540u, 2, 2, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_10, BASE_SCRATCHPAD0 + 0x3580u, 10, 1, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_11, BASE_SCRATCHPAD0 + 0x36C0u, 10, 1, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_11__reduce_dim0_in, BASE_SCRATCHPAD2 + 0x0u, 10, 1, 1, td, wd);
  // 修正：memref<1xi64> → (1,1,1)
  make_tensor(&tensor_alloc_12, BASE_SCRATCHPAD0 + 0x3940u, 1, 1, 1, td, wd);
  // 修正：memref<1xf32> → (1,1,1)
  make_tensor(&tensor_alloc_13, BASE_SCRATCHPAD0 + 0x3960u, 1, 1, 1, td, wd);
  // 修正：memref<1xf32> → (1,1,1)
  make_tensor(&tensor_alloc_14, BASE_SCRATCHPAD2 + 0x140u, 1, 1, 1, td, wd);
  // 修正：memref<1xi64> → (1,1,1)
  make_tensor(&tensor_alloc_15, BASE_SCRATCHPAD0 + 0x39A0u, 1, 1, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_16, BASE_SCRATCHPAD0 + 0x39C0u, 10, 1, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_17, BASE_SCRATCHPAD0 + 0x3B00u, 10, 1, 1, td, wd);
  // 修正：memref<1x1xf32> → (1,1,1)
  make_tensor(&tensor_alloc_18, BASE_SCRATCHPAD0 + 0x3C40u, 1, 1, 1, td, wd);
  // 修正：memref<1x1xf32> → (1,1,1)
  make_tensor(&tensor_alloc_19, BASE_SCRATCHPAD0 + 0x3C60u, 1, 1, 1, td, wd);
  // 修正：memref<1x1x14x14xf32> → (1,14,14)
  make_tensor(&tensor_alloc_2, BASE_SCRATCHPAD0 + 0x3C80u, 14, 14, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_20, BASE_SCRATCHPAD0 + 0x5500u, 10, 1, 1, td, wd);
  // 修正：memref<196x32xf32> → (32,196,1)
  make_tensor(&tensor_alloc_3, BASE_CIM0, 32, 196, 1, td, wd);
  // 修正：memref<1x32xf32> → (32,1,1)
  make_tensor(&tensor_alloc_4, BASE_SCRATCHPAD0 + 0x5640u, 32, 1, 1, td, wd);
  // 修正：memref<1x32xf32> → (32,1,1)
  make_tensor(&tensor_alloc_5, BASE_SCRATCHPAD0 + 0x5A40u, 32, 1, 1, td, wd);
  // 修正：memref<1x32xf32> → (32,1,1)
  make_tensor(&tensor_alloc_6, BASE_SCRATCHPAD0 + 0x5E40u, 32, 1, 1, td, wd);
  // 修正：memref<1x32xf32> → (32,1,1)
  make_tensor(&tensor_alloc_7, BASE_SCRATCHPAD0 + 0x6240u, 32, 1, 1, td, wd);
  // 修正：memref<32x10xf32> → (10,32,1)
  make_tensor(&tensor_alloc_8, BASE_CIM_FC, 10, 32, 1, td, wd);
  // 修正：memref<1x10xf32> → (10,1,1)
  make_tensor(&tensor_alloc_9, BASE_SCRATCHPAD0 + 0x6640u, 10, 1, 1, td, wd);
  // 修正：memref<1x196xf32> → (196,1,1)
  make_tensor(&tensor_collapse_shape, BASE_SCRATCHPAD0 + 0x3C80u, 196, 1, 1, td, wd);
  // 修正：memref<1x1xf32> → (1,1,1)
  make_tensor(&tensor_expand_shape, BASE_SCRATCHPAD0 + 0x6780u, 1, 1, 1, td, wd);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: flatten_view_operator
  make_tensor(&tensor_collapse_shape, tensor_alloc_2.base_addr, 196, 1, 1, tensor_alloc_2.type_data, tensor_alloc_2.wd_data);

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_4, &tensor_alloc_3, 0);

  // Call NPU operator: matmul_operator
  gemm_operator(&tensor_collapse_shape, &tensor_alloc_3, &tensor_alloc_5, &tensor_alloc_5, 0, 0);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_5, &tensor_0, &tensor_alloc_6, OPERATION_ADD);

  // Call NPU operator: relu_operator
  relu_operator(&tensor_alloc_6, &tensor_alloc_7);

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_12, &tensor_alloc_8, 0);

  // Call NPU operator: matmul_operator
  gemm_operator(&tensor_alloc_7, &tensor_alloc_8, &tensor_alloc_10, &tensor_alloc_10, 0, 0);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_10, &tensor_8, &tensor_alloc_11, OPERATION_ADD);

  // Call NPU operator: reduce_dim1_max
  reshape_operator(&tensor_alloc_11, &tensor_alloc_11__reduce_dim0_in);
  reduce_dim0_max(&tensor_alloc_11__reduce_dim0_in, &tensor_alloc_14);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_11, &tensor_expand_shape, &tensor_alloc_16, OPERATION_SUB);

  // Call NPU operator: lut_exp
  lut_exp(&tensor_alloc_16, &tensor_alloc_17);

  // Call NPU operator: reduce_dim1_sum
  reduce_dim1_sum(&tensor_alloc_17, &tensor_alloc_19);

  // Call NPU operator: div_operator
  // Broadcast div: reciprocal(divisor); stack Tensor view; tensor_vector MUL.
  lut_reciprocal(&tensor_alloc_19, &tensor_alloc_19);
  {
    Tensor __tv_in2_bc;
    memcpy(&__tv_in2_bc, &tensor_alloc_19, sizeof(Tensor));
    __tv_in2_bc.dim0 = tensor_alloc_17.dim0;
    __tv_in2_bc.dim1 = 1;
    __tv_in2_bc.dim2 = 1;
    __tv_in2_bc.byte_stride1 = tensor_alloc_17.byte_stride1;
    __tv_in2_bc.byte_stride2 = tensor_alloc_17.byte_stride2;
    __tv_in2_bc.type_data = tensor_alloc_17.type_data;
    __tv_in2_bc.wd_data = tensor_alloc_17.wd_data;
    tensor_vector_operator(&tensor_alloc_17, &__tv_in2_bc, &tensor_alloc_20, OPERATION_MUL);
  }

}
