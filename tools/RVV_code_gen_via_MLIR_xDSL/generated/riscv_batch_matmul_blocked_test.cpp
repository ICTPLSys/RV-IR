// Auto-generated C code from EmitC MLIR
// Using strategy: blocked
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <npu_highlevel.h>
#include <primitive.h>

// ====================================================================
// Tensor Helper Types and Functions
// ====================================================================


// Helper functions

// Helper function to compute minimum stride

// ====================================================================
// Generated function: forward
// Strategy: blocked
// ====================================================================

void *forward(void *arg0, void *arg1) {
  // Initialize NPU memory
  npu_mem_init();

  // Blocked GEMM strategy: tiling with block sizes (block_m=32, block_n=32, block_k=32)

  // Block sizes for tiling
  uint32_t block_m = 32;
  uint32_t block_n = 32;
  uint32_t block_k = 32;

   uint32_t start_addr = 0xa0000000;   // psram
  uint32_t spad_A_addr = 0x90000000;  // SPAD0 for A block
  uint32_t spad_B_addr = 0x80000;     // CIM for B block
  uint32_t spad_C_addr = 0x90020000;  // SPAD1 for C block

  // Tensor A: shape (batch=1, M=2048, K=128)
  int min_stride_A = min_stride1(128, WIDTH_8);
  Tensor tensor_A = (Tensor){
    .base_addr = -1,
    .dim0      = 128,
    .dim1      = 2048,
    .dim2      = 1,
    .byte_stride1 = min_stride_A,
    .byte_stride2 = min_stride_A * 2048,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Tensor B: shape (batch=1, M=512, K=2048)
  int min_stride_B = min_stride1(2048, WIDTH_8);
  Tensor tensor_B = (Tensor){
    .base_addr = -1,
    .dim0      = 2048,
    .dim1      = 512,
    .dim2      = 1,
    .byte_stride1 = min_stride_B,
    .byte_stride2 = min_stride_B * 512,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Tensor C: shape (batch=1, M=512, K=128)
  int min_stride_C = min_stride1(128, WIDTH_8);
  Tensor tensor_C = (Tensor){
    .base_addr = -1,
    .dim0      = 128,
    .dim1      = 512,
    .dim2      = 1,
    .byte_stride1 = min_stride_C,
    .byte_stride2 = min_stride_C * 512,
    .wd_data      = WIDTH_8,
    .type_data    = TYPE_INT
  };

  // Allocate memory in PSRAM for all matrices
  uint32_t tensor_size_A = getTensorSize(&tensor_A);
  uint32_t tensor_size_B = getTensorSize(&tensor_B);
  uint32_t tensor_size_C = getTensorSize(&tensor_C);

  tensor_A.base_addr = start_addr;
  tensor_B.base_addr = start_addr + tensor_size_A;
  tensor_C.base_addr = start_addr + tensor_size_A + tensor_size_B;

  // Loop over output blocks in C
  for (uint32_t bm = 0; bm < 2048; bm += block_m) {
    uint32_t cur_block_m = (bm + block_m <= 2048) ? block_m : (2048 - bm);

    for (uint32_t bn = 0; bn < 2048; bn += block_n) {
      uint32_t cur_block_n = (bn + block_n <= 2048) ? block_n : (2048 - bn);

      // Initialize accumulation for this output block
      uint32_t accumulate_flag = 0;

      int min_stride_C = min_stride1(cur_block_n, WIDTH_8);
      Tensor tensor_C_block = (Tensor){
        .base_addr = spad_C_addr,
        .dim0      = cur_block_n,
        .dim1      = cur_block_m,
        .dim2      = 1,
        .byte_stride1 = min_stride_C,
        .byte_stride2 = min_stride_C * cur_block_m,
        .wd_data      = WIDTH_8,
        .type_data    = TYPE_INT
      };

      // Loop over k dimension (reduction dimension)
      for (uint32_t bk = 0; bk < 128; bk += block_k) {
        uint32_t cur_block_k = (bk + block_k <= 128) ? block_k : (128 - bk);

        // Create block tensors
        int min_stride_A = min_stride1(cur_block_k, WIDTH_8);
        Tensor tensor_A_block = (Tensor){
          .base_addr = spad_A_addr,
          .dim0      = cur_block_k,
          .dim1      = cur_block_m,
          .dim2      = 1,
          .byte_stride1 = min_stride_A,
          .byte_stride2 = min_stride_A * cur_block_m,
          .wd_data      = WIDTH_8,
          .type_data    = TYPE_INT
        };

        int min_stride_B = min_stride1(cur_block_n, WIDTH_8);
        Tensor tensor_B_block = (Tensor){
          .base_addr = spad_B_addr,
          .dim0      = cur_block_n,
          .dim1      = cur_block_k,
          .dim2      = 1,
          .byte_stride1 = min_stride_B,
          .byte_stride2 = min_stride_B * cur_block_k,
          .wd_data      = WIDTH_8,
          .type_data    = TYPE_INT
        };

        // Create views of blocks from full tensors
        Tensor tensor_A_view = tensor_A;
        tensor_A_view.base_addr += bm * tensor_A.byte_stride1 + bk * 8;
        tensor_A_view.dim0 = cur_block_k;
        tensor_A_view.dim1 = cur_block_m;

        Tensor tensor_B_view = tensor_B;
        tensor_B_view.base_addr += bn * 8 + bk * tensor_B.byte_stride1;
        tensor_B_view.dim0 = cur_block_n;
        tensor_B_view.dim1 = cur_block_k;

        Tensor tensor_C_view = tensor_C;
        tensor_C_view.base_addr += bm * tensor_C.byte_stride1 + bn * 8;
        tensor_C_view.dim0 = cur_block_n;
        tensor_C_view.dim1 = cur_block_m;

        // Load blocks from PSRAM to scratchpad
        tensor_load(&tensor_A_view, &tensor_A_block);
        tensor_load(&tensor_B_view, &tensor_B_block);

        // Perform block GEMM
        gemm_operator(&tensor_A_block, &tensor_B_block, &tensor_C_block, &tensor_C_block, accumulate_flag, 0);

        accumulate_flag = 1;
      }

      // Store final C block back to PSRAM
      Tensor tensor_C_view_final = tensor_C;
      tensor_C_view_final.base_addr += bm * tensor_C.byte_stride1 + bn * 8;
      tensor_C_view_final.dim0 = cur_block_n;
      tensor_C_view_final.dim1 = cur_block_m;

      tensor_store(&tensor_C_block, &tensor_C_view_final);
    }
  }

  return (void *)tensor_C.base_addr;
}

// ====================================================================
// Main Function (for testing)
// ====================================================================

int main(void) {
  forward(NULL, NULL);
    return 0;
}
