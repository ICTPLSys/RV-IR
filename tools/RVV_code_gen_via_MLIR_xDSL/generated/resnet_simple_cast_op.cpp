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

  // ====================================================================
  // Tensor buffers (make_tensor fills layout)
  // ====================================================================

  Tensor tensor_arg0;
  Tensor tensor_alloc;
  Tensor tensor_alloc_i8;
  Tensor tensor_bn_beta;
  Tensor tensor_bn_gamma;
  Tensor tensor_bn_mean;
  Tensor tensor_bn_out;
  Tensor tensor_bn_var;
  Tensor tensor_collapsed;
  Tensor tensor_collapsed_i8;
  Tensor tensor_conv_wrk;
  Tensor tensor_conv_wrk_i8;
  Tensor tensor_fc_b;
  Tensor tensor_fc_w;
  Tensor tensor_fc_w_i8;
  Tensor tensor_fc_w_t;
  Tensor tensor_fc_w_t_i8;
  Tensor tensor_in;
  Tensor tensor_logits;
  Tensor tensor_logits_i8;
  Tensor tensor_out;
  Tensor tensor_pool_k;
  Tensor tensor_pooled_avg;
  Tensor tensor_pooled_wrk;
  Tensor tensor_relu_out;
  Tensor tensor_subview;
  Tensor tensor_w_conv;

  // SPAD / CIM placement: BASE_SCRATCHPAD0..3 only (test_case_i8_conv_convert.c); subviews use parent base_addr + byte offset.

  make_tensor(&tensor_arg0, BASE_SCRATCHPAD3 + 0x0u, 3, 4, 4, td, wd);
  make_tensor(&tensor_alloc, BASE_SCRATCHPAD0 + 0x0u, 3, 10, 10, td, wd);
  make_tensor(&tensor_alloc_i8, BASE_SCRATCHPAD0 + 0x2000u, 3, 10, 10, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_bn_beta, BASE_SCRATCHPAD3 + 0x200u, 4, 1, 1, td, wd);
  make_tensor(&tensor_bn_gamma, BASE_SCRATCHPAD3 + 0x220u, 4, 1, 1, td, wd);
  make_tensor(&tensor_bn_mean, BASE_SCRATCHPAD3 + 0x240u, 4, 1, 1, td, wd);
  make_tensor(&tensor_bn_out, BASE_SCRATCHPAD0 + 0xC80u, 4, 8, 8, td, wd);
  make_tensor(&tensor_bn_var, BASE_SCRATCHPAD3 + 0x260u, 4, 1, 1, td, wd);
  make_tensor(&tensor_collapsed, BASE_SCRATCHPAD0 + 0x1480u, 4, 1, 1, td, wd);
  make_tensor(&tensor_collapsed_i8, BASE_SCRATCHPAD0 + 0x2C80u, 4, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_conv_wrk, BASE_SCRATCHPAD1 + 0x0u, 4, 8, 8, td, wd);
  make_tensor(&tensor_conv_wrk_i8, BASE_SCRATCHPAD1 + 0x1800u, 4, 8, 8, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_fc_b, BASE_SCRATCHPAD3 + 0x280u, 2, 1, 1, td, wd);
  make_tensor(&tensor_fc_w, BASE_SCRATCHPAD2 + 0x0u, 4, 2, 1, td, wd);
  make_tensor(&tensor_fc_w_i8, BASE_SCRATCHPAD2 + 0x200u, 4, 2, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_fc_w_t, BASE_CIM_FC, 2, 4, 1, td, wd);
  make_tensor(&tensor_fc_w_t_i8, BASE_CIM_FC, 2, 4, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_in, BASE_SCRATCHPAD3 + 0x2A0u, 3, 4, 4, td, wd);
  make_tensor(&tensor_logits, BASE_SCRATCHPAD0 + 0x14A0u, 2, 1, 1, td, wd);
  make_tensor(&tensor_logits_i8, BASE_SCRATCHPAD0 + 0x2CA0u, 2, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_out, BASE_SCRATCHPAD1 + 0x800u, 2, 1, 1, td, wd);
  make_tensor(&tensor_pool_k, BASE_SCRATCHPAD2 + 0x40u, 8, 8, 1, td, wd);
  make_tensor(&tensor_pooled_avg, BASE_SCRATCHPAD0 + 0x14C0u, 4, 1, 1, td, wd);
  make_tensor(&tensor_pooled_wrk, BASE_SCRATCHPAD3 + 0x4A0u, 4, 1, 1, td, wd);
  make_tensor(&tensor_relu_out, BASE_SCRATCHPAD1 + 0x820u, 4, 8, 8, td, wd);
  make_tensor(&tensor_subview, tensor_alloc.base_addr + 3u * (uint32_t)tensor_alloc.byte_stride2 + 3u * (uint32_t)tensor_alloc.byte_stride1, 3, 4, 4, td, wd);
  tensor_subview.byte_stride1 = tensor_alloc.byte_stride1;
  tensor_subview.byte_stride2 = tensor_alloc.byte_stride2;
  make_tensor(&tensor_w_conv, BASE_CIM0, 4, 3, 9, td, wd);

  // constantofshape on working tensors (test_case_i8_conv_convert.c)
  constantofshape_operator(&tensor_alloc, 0u);
  constantofshape_operator(&tensor_alloc_i8, 0u);
  constantofshape_operator(&tensor_conv_wrk, 0u);
  constantofshape_operator(&tensor_conv_wrk_i8, 0u);
  constantofshape_operator(&tensor_bn_out, 0u);
  constantofshape_operator(&tensor_relu_out, 0u);
  constantofshape_operator(&tensor_pooled_wrk, 0u);
  constantofshape_operator(&tensor_pooled_avg, 0u);
  constantofshape_operator(&tensor_logits, 0u);
  constantofshape_operator(&tensor_logits_i8, 0u);
  constantofshape_operator(&tensor_out, 0u);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // MLIR copy into memref.subview: row-wise tensor_tensor_operator(..., OPERATION_ADD); parent strides for dst rows (see memref.subview vs compact src).
  constantofshape_operator(&tensor_alloc, 0u);
  constantofshape_operator(&tensor_subview, 0u);
  for (int row = 0; row < 4; ++row) {
    Tensor row_src;
    Tensor row_dst;
    make_tensor(&row_src, tensor_arg0.base_addr + (uint32_t)row * (uint32_t)tensor_arg0.byte_stride2, 3, 1, 4, td, wd);
    make_tensor(&row_dst, tensor_subview.base_addr + (uint32_t)row * (uint32_t)tensor_alloc.byte_stride2, 3, 1, 4, td, wd);
    tensor_tensor_operator(&row_src, &row_dst, &row_dst, OPERATION_ADD);
  }

  // Cast conv input FP16 -> INT8 (conv primitive currently supports INT8 path)
  cast_operator(&tensor_alloc, &tensor_alloc_i8);

  // Call NPU operator: conv_operator
  {
    for (int kx = 0; kx < 3; ++kx) {
      Tensor shifted;
      memcpy(&shifted, &tensor_alloc_i8, sizeof(Tensor));
      shifted.dim1 = tensor_conv_wrk_i8.dim1;
      shifted.dim2 = tensor_conv_wrk_i8.dim2 + 3 - 1;
      shifted.base_addr = tensor_alloc_i8.base_addr + (uint32_t)kx * (uint32_t)tensor_alloc_i8.byte_stride1;
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
      conv_operator(&shifted, &tensor_conv_wrk_i8, &tensor_conv_wrk_i8, &conv_opt);
    }
  }
  // Cast conv output INT8 -> FP16 for following FP16 tensor-vector ops
  cast_operator(&tensor_conv_wrk_i8, &tensor_conv_wrk);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_conv_wrk, &tensor_bn_mean, &tensor_bn_out, OPERATION_SUB);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_bn_out, &tensor_bn_gamma, &tensor_bn_out, OPERATION_MUL);

  // Call NPU operator: tensor_vector_operator
  tensor_vector_operator(&tensor_bn_out, &tensor_bn_beta, &tensor_bn_out, OPERATION_ADD);

  // Call NPU operator: relu_operator
  relu_operator(&tensor_bn_out, &tensor_relu_out);

  // Call NPU operator: pooling_nchw_sum
  reduce_dim2_dim1_sum(&tensor_relu_out, &tensor_pooled_wrk);

  // Call NPU operator: tensor_imm_operator
  // FP16 scalar 1/64 uses raw half bits 0x2400.
  tensor_imm_operator(&tensor_pooled_wrk, &tensor_pooled_avg, 0x2400u, WIDTH_16, TYPE_FP, OPERATION_MUL);

  // Call NPU operator: flatten_view_operator
  make_tensor(&tensor_collapsed, tensor_pooled_avg.base_addr, 4, 1, 1, tensor_pooled_avg.type_data, tensor_pooled_avg.wd_data);

  // Cast GEMM inputs FP16 -> INT8, then run INT8 transpose/GEMM path
  cast_operator(&tensor_fc_w, &tensor_fc_w_i8);
  cast_operator(&tensor_collapsed, &tensor_collapsed_i8);

  // Call NPU operator: transpose_operator (INT8)
  transpose_operator(&tensor_fc_w_i8, &tensor_fc_w_t_i8, 0);

  // Call NPU operator: matmul_operator (INT8)
  gemm_operator(&tensor_collapsed_i8, &tensor_fc_w_t_i8, &tensor_logits_i8, &tensor_logits_i8, 0, 0);
  // Cast GEMM output INT8 -> FP16 for final FP16 add
  cast_operator(&tensor_logits_i8, &tensor_logits);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_logits, &tensor_fc_b, &tensor_out, OPERATION_ADD);

}
