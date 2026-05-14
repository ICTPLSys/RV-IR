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
// Generated function: test_reduce_min
// ====================================================================

void test_reduce_min(void) {
  // Initialize NPU memory
  npu_mem_init();

  const int td = TYPE_INT;
  const int wd = WIDTH_8;

  // ====================================================================
  // Tensor buffers (make_tensor fills layout)
  // ====================================================================

  Tensor tensor_arg0;
  Tensor tensor_alloc;

  // Reduce path: pack activations/scratch in SPAD0..SPAD3 (SPAD_BANK_BYTES per bank, SPAD_OFFSET_ALIGN-byte offsets). CIM weights unchanged.

  make_tensor(&tensor_arg0, BASE_SCRATCHPAD0 + 0x0u, 2048, 128, 1, td, wd);
  make_tensor(&tensor_alloc, BASE_SCRATCHPAD2 + 0x0u, 1, 128, 1, td, wd);

  // ====================================================================
  // NPU Operator Calls
  // ====================================================================

  // Call NPU operator: reduce_dim0_min
  reduce_dim0_min(&tensor_arg0, &tensor_alloc);

}
