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

  const int td = TYPE_FP;
  const int wd = WIDTH_16;
  // linalg_mlir_to_c: tensor_type='fp', wd_bits=16, int8_cast_around_matmul=yes

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
  Tensor tensor_12_i8;
  Tensor tensor_4_i8;
  Tensor tensor_alloc_10_i8;
  Tensor tensor_alloc_3_i8;
  Tensor tensor_alloc_5_i8;
  Tensor tensor_alloc_7_i8;
  Tensor tensor_alloc_8_i8;
  Tensor tensor_collapse_shape_i8;

  // Reduce path: pack activations/scratch in SPAD0..SPAD3 (SPAD_BANK_BYTES per bank, SPAD_OFFSET_ALIGN-byte offsets). CIM weights unchanged.

  make_tensor(&tensor_0, BASE_SCRATCHPAD0 + 0x0u, 32, 1, 1, td, wd);
  make_tensor(&tensor_12, BASE_SCRATCHPAD0 + 0x40u, 32, 10, 1, td, wd);
  make_tensor(&tensor_4, BASE_SCRATCHPAD0 + 0x2C0u, 196, 32, 1, td, wd);
  make_tensor(&tensor_8, BASE_SCRATCHPAD0 + 0x36C0u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc, BASE_SCRATCHPAD0 + 0x36E0u, 14, 14, 1, td, wd);
  make_tensor(&tensor_alloc_1, BASE_SCRATCHPAD0 + 0x38A0u, 2, 2, 1, td, wd);
  make_tensor(&tensor_alloc_10, BASE_SCRATCHPAD0 + 0x38E0u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_11, BASE_SCRATCHPAD0 + 0x3900u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_11__reduce_dim0_in, BASE_SCRATCHPAD2 + 0x0u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_12, BASE_SCRATCHPAD0 + 0x3940u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_13, BASE_SCRATCHPAD0 + 0x3960u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_14, BASE_SCRATCHPAD2 + 0x20u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_15, BASE_SCRATCHPAD0 + 0x39A0u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_16, BASE_SCRATCHPAD0 + 0x39C0u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_17, BASE_SCRATCHPAD0 + 0x39E0u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_18, BASE_SCRATCHPAD0 + 0x3A00u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_19, BASE_SCRATCHPAD2 + 0x40u, 1, 1, 1, td, wd);
  make_tensor(&tensor_alloc_2, BASE_SCRATCHPAD0 + 0x3A40u, 14, 14, 1, td, wd);
  make_tensor(&tensor_alloc_20, BASE_SCRATCHPAD0 + 0x3C00u, 10, 1, 1, td, wd);
  make_tensor(&tensor_alloc_3, BASE_CIM0, 32, 196, 1, td, wd);
  make_tensor(&tensor_alloc_4, BASE_SCRATCHPAD0 + 0x3C20u, 32, 1, 1, td, wd);
  make_tensor(&tensor_alloc_5, BASE_SCRATCHPAD0 + 0x3C60u, 32, 1, 1, td, wd);
  make_tensor(&tensor_alloc_6, BASE_SCRATCHPAD0 + 0x3CA0u, 32, 1, 1, td, wd);
  make_tensor(&tensor_alloc_7, BASE_SCRATCHPAD0 + 0x3CE0u, 32, 1, 1, td, wd);
  make_tensor(&tensor_alloc_8, (BASE_CIM_FC + CIM_PAGE_BYTES), 10, 32, 1, td, wd);
  make_tensor(&tensor_alloc_9, BASE_SCRATCHPAD0 + 0x3D20u, 10, 1, 1, td, wd);
  make_tensor(&tensor_collapse_shape, BASE_SCRATCHPAD0 + 0x3A40u, 196, 1, 1, td, wd);
  make_tensor(&tensor_expand_shape, BASE_SCRATCHPAD0 + 0x3D40u, 10, 1, 1, td, wd);
  make_tensor(&tensor_12_i8, BASE_SCRATCHPAD0 + 0x3D60u, 32, 10, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_4_i8, BASE_SCRATCHPAD0 + 0x3EA0u, 196, 32, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_10_i8, BASE_SCRATCHPAD0 + 0x5AA0u, 10, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_3_i8, BASE_CIM0, 32, 196, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_5_i8, BASE_SCRATCHPAD0 + 0x5AC0u, 32, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_7_i8, BASE_SCRATCHPAD0 + 0x5AE0u, 32, 1, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_alloc_8_i8, (BASE_CIM_FC + CIM_PAGE_BYTES), 10, 32, 1, TYPE_INT, WIDTH_8);
  make_tensor(&tensor_collapse_shape_i8, BASE_SCRATCHPAD0 + 0x5B00u, 196, 1, 1, TYPE_INT, WIDTH_8);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: flatten_view_operator
  make_tensor(&tensor_collapse_shape, tensor_alloc_2.base_addr, 196, 1, 1, tensor_alloc_2.type_data, tensor_alloc_2.wd_data);

  // Call NPU operator: transpose_operator
  cast_operator(&tensor_4, &tensor_4_i8);
  transpose_operator(&tensor_4_i8, &tensor_alloc_3_i8, 0);

  // Call NPU operator: matmul_operator
  cast_operator(&tensor_collapse_shape, &tensor_collapse_shape_i8);
  gemm_operator(&tensor_collapse_shape_i8, &tensor_alloc_3_i8, &tensor_alloc_5_i8, &tensor_alloc_5_i8, 0, 0);
  cast_operator(&tensor_alloc_5_i8, &tensor_alloc_5);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_5, &tensor_0, &tensor_alloc_6, OPERATION_ADD);

  // Call NPU operator: relu_operator
  relu_operator(&tensor_alloc_6, &tensor_alloc_7);

  // Call NPU operator: transpose_operator
  cast_operator(&tensor_12, &tensor_12_i8);
  transpose_operator(&tensor_12_i8, &tensor_alloc_8_i8, 0);

  // Call NPU operator: matmul_operator
  cast_operator(&tensor_alloc_7, &tensor_alloc_7_i8);
  gemm_operator(&tensor_alloc_7_i8, &tensor_alloc_8_i8, &tensor_alloc_10_i8, &tensor_alloc_10_i8, 0, 0);
  cast_operator(&tensor_alloc_10_i8, &tensor_alloc_10);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_10, &tensor_8, &tensor_alloc_11, OPERATION_ADD);

  // Call NPU operator: reduce_dim1_max
  reshape_operator(&tensor_alloc_11, &tensor_alloc_11__reduce_dim0_in);
  reduce_dim0_max(&tensor_alloc_11__reduce_dim0_in, &tensor_alloc_14);

  // Call NPU operator: tensor_tensor_operator
  tensor_tensor_operator(&tensor_alloc_11, &tensor_expand_shape, &tensor_alloc_16, OPERATION_SUB);

  // Call NPU operator: lut_exp
  lut_exp(&tensor_alloc_16, &tensor_alloc_17);

  // Call NPU operator: reduce_dim0_sum
  reduce_dim0_sum(&tensor_alloc_17, &tensor_alloc_19);

  // Call NPU operator: div_operator
  // Broadcast div: lut_reciprocal(scalar); vp_cfg_val_in2 + vp_drv_vs_v (SDK softmax tail).
  lut_reciprocal(&tensor_alloc_19, &tensor_alloc_19);
  {
    Tensor *__vs_in1 = &tensor_alloc_17;
    Tensor *__vs_sclr = &tensor_alloc_19;
    Tensor *__vs_out = &tensor_alloc_20;
    vp_cfg_push();
    vp_cfg_type(__vs_in1->type_data, __vs_in1->type_data, __vs_out->type_data);
    vp_cfg_shape(
        __vs_in1->dim0,
        __vs_in1->wd_data,
        __vs_in1->wd_data,
        __vs_out->wd_data
    );
    vp_cfg_op(OPERATION_MUL);
    for (int __d2 = 0; __d2 < __vs_in1->dim2; __d2++) {
      uint32_t __b_in = __vs_in1->base_addr + __d2 * __vs_in1->byte_stride2;
      uint32_t __b_out = __vs_out->base_addr + __d2 * __vs_out->byte_stride2;
      for (int __d1 = 0; __d1 < __vs_in1->dim1; __d1++) {
        uint32_t __imm = (__vs_sclr->wd_data == WIDTH_16) ? (ctrl_rd_mem_2b(__vs_sclr->base_addr) & 0xffffu) : ctrl_rd_mem(__vs_sclr->base_addr);
        vp_cfg_val_in2(__imm);
        vp_drv_vs_v(__b_in, __b_out);
        __b_in += __vs_in1->byte_stride1;
        __b_out += __vs_out->byte_stride1;
      }
    }
    vp_cfg_pop();
  }

}
