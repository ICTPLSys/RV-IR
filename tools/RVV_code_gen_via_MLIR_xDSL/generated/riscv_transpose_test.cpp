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

// ====================================================================
// Generated function: transpose_test
// ====================================================================

void transpose_test(void) {
  // Initialize NPU memory
  npu_mem_init();

  const int td = TYPE_INT;
  const int wd = WIDTH_16;

  // ====================================================================
  // Tensor buffers (make_tensor fills layout)
  // ====================================================================

  Tensor tensor_alloc_input;
  Tensor tensor_alloc_output;

  // SPAD / CIM placement: BASE_SCRATCHPAD0..3 only (test_case_i8_conv_convert.c); subviews use parent base_addr + byte offset.

  // Align with run_simulator_trans.py:
  // input tensor shape = [64, 10, 20] stored at 0x9001_0000 (SPAD0 + 0x10000)
  // output tensor shape = [20, 10, 64] stored at 0x9005_0000 (SPAD2 + 0x10000)
  // For Tensor fields, we use (dim0, dim1, dim2) = (C, W, H),
  // so transpose_operator swaps dim0 <-> dim1 and keeps dim2.
  make_tensor(&tensor_alloc_input, BASE_SCRATCHPAD0 + 0x10000u, 64, 20, 10, td, wd);
  make_tensor(&tensor_alloc_output, BASE_SCRATCHPAD2 + 0x10000u, 20, 64, 10, td, wd);

  // constantofshape on working tensors (test_case_i8_conv_convert.c)

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: transpose_operator
  transpose_operator(&tensor_alloc_input, &tensor_alloc_output, 0);

}
