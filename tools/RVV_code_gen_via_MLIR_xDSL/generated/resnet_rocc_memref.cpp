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
#define CIM_CONV_PARTS 3u
#define BASE_CIM_FC (BASE_CIM0 + CIM_CONV_PARTS * CIM_PAGE_BYTES)

extern int conv_operator(Tensor *tensor_in, Tensor *tensor_out, Tensor *tensor_orig, CONV_OPTION *conv_option);

extern int tensor_vector_operator(Tensor *tensor_in1, Tensor *tensor_in2, Tensor *tensor_out, uint32_t tensor_op);

extern int relu_operator(Tensor *tensor_in, Tensor *tensor_out);

// ====================================================================
// Generated function: forward
// ====================================================================

void forward(void) {
  // Initialize NPU memory
  npu_mem_init();

  const int td = TYPE_FP;
  const int wd = WIDTH_16;
  // linalg_mlir_to_c: tensor_type='fp', wd_bits=16, int8_cast_around_matmul=yes

  // ====================================================================
  // Tensor buffers (make_tensor fills layout)
  // ====================================================================

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
  Tensor tensor_alloc_2;
  Tensor tensor_alloc_3;
  Tensor tensor_alloc_4;
  Tensor tensor_alloc_5;
  Tensor tensor_alloc_6;
  Tensor tensor_alloc_7;
  Tensor tensor_alloc_8;
  Tensor tensor_alloc_9;
  Tensor tensor_collapse_shape;
  Tensor tensor_subview;
  Tensor tensor_alloc_16_i8;
  Tensor tensor_alloc_5_i8;
  Tensor tensor_alloc_6_i8;
  Tensor tensor_alloc_8_i8;
  Tensor tensor_collapse_shape_i8;

  // SPAD / CIM placement: BASE_SCRATCHPAD0..3 only (test_case_i8_conv_convert.c); subviews use parent base_addr + byte offset.

  make_tensor(&tensor_alloc, BASE_SCRATCHPAD0 + 0x0u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_1, BASE_CIM0, 4, 3, 9, td, wd);
  make_tensor(&tensor_alloc_10, BASE_SCRATCHPAD3 + 0x0u, 4, 9, 9, td, wd);
  make_tensor(&tensor_alloc_11, BASE_SCRATCHPAD0 + 0x20u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_12, BASE_SCRATCHPAD1 + 0x0u, 9, 9, 1, td, wd);
  make_tensor(&tensor_alloc_13, BASE_SCRATCHPAD2 + 0x0u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_14, BASE_SCRATCHPAD3 + 0xA20u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_15, BASE_SCRATCHPAD0 + 0x40u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_16, BASE_SCRATCHPAD1 + 0x120u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_17, BASE_SCRATCHPAD2 + 0x20u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_2, BASE_SCRATCHPAD0 + 0x60u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_3, BASE_SCRATCHPAD1 + 0x140u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_4, BASE_SCRATCHPAD2 + 0x40u, 4, 1, 1, td, wd);
  make_tensor(&tensor_alloc_5, BASE_CIM_FC, 4, 4, 1, td, wd);
  make_tensor(&tensor_alloc_6, BASE_SCRATCHPAD0 + 0x80u, 3, 11, 11, td, wd);
  make_tensor(&tensor_alloc_7, BASE_SCRATCHPAD1 + 0x160u, 4, 9, 9, td, wd);
  make_tensor(&tensor_alloc_8, BASE_SCRATCHPAD2 + 0x60u, 4, 9, 9, td, wd);
  make_tensor(&tensor_alloc_9, BASE_SCRATCHPAD3 + 0xA40u, 4, 9, 9, td, wd);
  make_tensor(&tensor_collapse_shape, BASE_SCRATCHPAD3 + 0x1460u, 4, 1, 1, td, wd);
  make_tensor(&tensor_subview, tensor_alloc_6.base_addr + 3u * (uint32_t)tensor_alloc_6.byte_stride2 + 3u * (uint32_t)tensor_alloc_6.byte_stride1, 3, 5, 5, td, wd);
  tensor_subview.byte_stride1 = tensor_alloc_6.byte_stride1;
  tensor_subview.byte_stride2 = tensor_alloc_6.byte_stride2;
  make_tensor(&tensor_alloc_16_i8, BASE_SCRATCHPAD1 + 0x1800u, 4, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_5_i8, BASE_CIM_FC, 4, 4, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_6_i8, BASE_SCRATCHPAD0 + 0x2000u, 3, 11, 11, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_8_i8, BASE_SCRATCHPAD2 + 0xA80u, 4, 9, 9, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_collapse_shape_i8, BASE_SCRATCHPAD3 + 0x1480u, 4, 1, 1, TYPE_INT, WIDTH_8);

  // constantofshape on working tensors (test_case_i8_conv_convert.c)
  constantofshape_operator(&tensor_alloc, 0u);
  constantofshape_operator(&tensor_alloc_16_i8, 0u);
  /* tensor_alloc_5_i8: CIM weight — do not constantofshape (host fills CIM). */
  constantofshape_operator(&tensor_alloc_6_i8, 0u);
  constantofshape_operator(&tensor_alloc_8_i8, 0u);
  constantofshape_operator(&tensor_collapse_shape_i8, 0u);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: conv_operator
  cast_operator(&tensor_alloc_6, &tensor_alloc_6_i8);
  {
    for (int kx = 0; kx < 3; ++kx) {
      Tensor shifted;
      memcpy(&shifted, &tensor_alloc_6_i8, sizeof(Tensor));
      shifted.dim1 = tensor_alloc_8_i8.dim1;
      shifted.dim2 = tensor_alloc_8_i8.dim2 + 3 - 1;
      shifted.base_addr = tensor_alloc_6_i8.base_addr + (uint32_t)kx * (uint32_t)tensor_alloc_6_i8.byte_stride1;
      CONV_OPTION conv_opt;
      memset(&conv_opt, 0, sizeof(conv_opt));
      conv_opt.type_data = (uint32_t)TYPE_INT;
      conv_opt.wd_data = (uint32_t)WIDTH_8;
      conv_opt.byte_base_wt = BASE_CIM0 + (uint32_t)kx * CIM_PAGE_BYTES;
      conv_opt.accumulate = (kx == 0) ? 0u : 1u;
      conv_opt.activate = 0u;
      conv_opt.shift = 0u;
      conv_opt.size_x = 1u;
      conv_opt.size_y = 3u;
      conv_opt.slide_x = 1u; conv_opt.slide_y = 1u;
      conv_opt.dilate_x = 1u; conv_opt.dilate_y = 1u;
      conv_opt.log2trs_x = 0u; conv_opt.log2trs_y = 0u;
      conv_opt.padding_w = 0u; conv_opt.padding_n = 0u;
      conv_opt.padding_value = 0u;
      conv_operator(&shifted, &tensor_alloc_8_i8, &tensor_alloc_8_i8, &conv_opt);
    }
  }
  cast_operator(&tensor_alloc_8_i8, &tensor_alloc_8);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_alloc_8, &tensor_alloc_3, &tensor_alloc_9, OPERATION_SUB);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_alloc_9, &tensor_alloc_2, &tensor_alloc_9, OPERATION_MUL);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_alloc_9, &tensor_alloc_3, &tensor_alloc_9, OPERATION_ADD);

  // Call NPU operator: relu_operator
  relu_operator(&tensor_alloc_9, &tensor_alloc_10);

  // Call NPU operator: pooling_nchw_sum
  reduce_dim2_dim1_sum(&tensor_alloc_10, &tensor_alloc_13);

  // Call NPU operator: tensor_imm_operator
  tensor_imm_operator(&tensor_alloc_13, &tensor_alloc_14, 0x2252u, WIDTH_16, TYPE_FP, OPERATION_MUL);

  // Call NPU operator: flatten_view_operator
  make_tensor(&tensor_collapse_shape, tensor_alloc_14.base_addr, 4, 1, 1, tensor_alloc_14.type_data, tensor_alloc_14.wd_data);

  // Call NPU operator: matmul_operator
  cast_operator(&tensor_collapse_shape, &tensor_collapse_shape_i8);
  gemm_operator(&tensor_collapse_shape_i8, &tensor_alloc_5_i8, &tensor_alloc_16_i8, &tensor_alloc_16_i8, 0, 0);
  cast_operator(&tensor_alloc_16_i8, &tensor_alloc_16);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_16, &tensor_alloc_4, &tensor_alloc_17, OPERATION_ADD);

}
