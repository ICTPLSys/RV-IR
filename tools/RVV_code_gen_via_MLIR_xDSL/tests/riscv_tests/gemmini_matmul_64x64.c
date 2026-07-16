// Auto-generated Gemmini baremetal program from RAIR MLIR
// matmul: C[64x64] = A[64x32] * B[32x64]

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define MAT_DIM_I 64
#define MAT_DIM_K 32
#define MAT_DIM_J 64

static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
static elem_t C_gold[MAT_DIM_I][MAT_DIM_J];

static elem_t saturate(full_t x) {
  #ifndef ELEM_T_IS_FLOAT
  if (x > elem_t_max) return elem_t_max;
  if (x < elem_t_min) return elem_t_min;
  #endif
  return (elem_t)x;
}

int main(void) {
  #ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
  #endif

  printf("Gemmini matmul: C[%d,%d] = A[%d,%d] * B[%d,%d]\n",
      MAT_DIM_I, MAT_DIM_J, MAT_DIM_I, MAT_DIM_K, MAT_DIM_K, MAT_DIM_J);

  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t k = 0; k < MAT_DIM_K; k++)
      A[i][k] = (elem_t)((i + 2 * k) % 5 - 2);

  for (size_t k = 0; k < MAT_DIM_K; k++)
    for (size_t j = 0; j < MAT_DIM_J; j++)
      B[k][j] = (elem_t)((3 * k + j) % 7 - 3);

  for (size_t i = 0; i < MAT_DIM_I; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++)
      C[i][j] = 0;

  uint64_t cpu_start = read_cycles();
  for (size_t i = 0; i < MAT_DIM_I; i++) {
    for (size_t j = 0; j < MAT_DIM_J; j++) {
      full_t sum = 0;
      for (size_t k = 0; k < MAT_DIM_K; k++)
        sum += (full_t)A[i][k] * (full_t)B[k][j];
      C_gold[i][j] = saturate(sum);
    }
  }
  uint64_t cpu_end = read_cycles();
  printf("CPU reference took %llu cycles\n", cpu_end - cpu_start);

  gemmini_flush(0);

  uint64_t start = read_cycles();

  // Tiled matmul (64x32 * 32x64), DIM=16
  // Tile grid: I=4, J=4, K=2  (pad_I=0, pad_J=0, pad_K=0)
  // Following sp_tiled_matmul_os addressing: A from bottom, B from top of scratchpad

  const size_t I = 4;
  const size_t J = 4;
  const size_t K_tiles = 2;

  // Scratchpad address layout (matches sp_tiled_matmul_os)
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = 4 * 4096 - K_tiles * J * DIM;
  const uint32_t C_sp_addr_start = (3u << (32 - 2));

  gemmini_config_ex(OUTPUT_STATIONARY, 0, 0);

  // ---- Move-in B (all tiles) ----
  gemmini_config_ld(MAT_DIM_J * sizeof(elem_t));
  for (size_t k = 0; k < K_tiles; k++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;
      size_t cols = DIM - (j == J - 1 ? 0 : 0);
      size_t rows = DIM - (k == K_tiles - 1 ? 0 : 0);
      gemmini_extended_mvin(&B[k * DIM][j * DIM], B_sp_addr, cols, rows);
    }
  }

  // ---- Move-in A (all tiles) ----
  gemmini_config_ld(MAT_DIM_K * sizeof(elem_t));
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K_tiles; k++) {
      const uint32_t A_sp_addr = A_sp_addr_start + (i * K_tiles + k) * DIM;
      size_t cols = DIM - (k == K_tiles - 1 ? 0 : 0);
      size_t rows = DIM - (i == I - 1 ? 0 : 0);
      gemmini_extended_mvin(&A[i * DIM][k * DIM], A_sp_addr, cols, rows);
    }
  }

  // ---- Compute C = A * B (output-stationary) ----
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;

      for (size_t k = 0; k < K_tiles; k++) {
        const uint32_t A_sp_addr = A_sp_addr_start + (i * K_tiles + k) * DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;

        size_t A_cols = DIM - (k == K_tiles - 1 ? 0 : 0);
        size_t A_rows = DIM - (i == I - 1 ? 0 : 0);
        size_t B_cols = DIM - (j == J - 1 ? 0 : 0);
        size_t B_rows = DIM - (k == K_tiles - 1 ? 0 : 0);
        size_t C_cols = DIM - (j == J - 1 ? 0 : 0);
        size_t C_rows = DIM - (i == I - 1 ? 0 : 0);

        // Last k-tile outputs to C_sp_addr; others to GARBAGE_ADDR
        uint32_t out_sp_addr = (k == K_tiles - 1) ? C_sp_addr : GARBAGE_ADDR;

        gemmini_extended_preload(GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);

        if (k == 0) {
          gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        } else {
          gemmini_extended_compute_accumulated(A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
        }
      }  // k
    }  // j
  }  // i

  // ---- Move-out C (all tiles) ----
  gemmini_config_st(MAT_DIM_J * sizeof(elem_t));
  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;
      size_t C_cols = DIM - (j == J - 1 ? 0 : 0);
      size_t C_rows = DIM - (i == I - 1 ? 0 : 0);
      gemmini_extended_mvout(&C[i * DIM][j * DIM], C_sp_addr, C_cols, C_rows);
    }
  }

  gemmini_fence();

  uint64_t end = read_cycles();
  printf("Gemmini matmul took %llu cycles\n", end - start);

  int pass = 1;
  for (size_t i = 0; i < MAT_DIM_I; i++) {
    for (size_t j = 0; j < MAT_DIM_J; j++) {
      if (C[i][j] != C_gold[i][j]) {
        printf("MISMATCH at C[%u][%u]: got %d, expected %d\n",
            (unsigned)i, (unsigned)j, C[i][j], C_gold[i][j]);
        pass = 0;
      }
    }
  }

  if (pass) {
    printf("PASSED\n");
    exit(0);
  } else {
    printf("FAILED\n");
    exit(1);
  }
}
