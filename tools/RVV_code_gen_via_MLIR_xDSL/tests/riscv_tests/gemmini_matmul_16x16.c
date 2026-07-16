// Auto-generated Gemmini baremetal program from RAIR MLIR
// matmul: C[16x16] = A[16x16] * B[16x16]

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define MAT_DIM_I 16
#define MAT_DIM_K 16
#define MAT_DIM_J 16

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

  // Single-tile matmul (16x16 * 16x16), fits in DIM=16

  // Scratchpad addresses
  size_t A_sp_addr = 0;
  size_t B_sp_addr = DIM;
  size_t C_sp_addr = 2 * DIM;

  // Configure load stride and move A, B into scratchpad
  gemmini_config_ld(MAT_DIM_K * sizeof(elem_t));
  gemmini_mvin(A, A_sp_addr);

  gemmini_config_ld(MAT_DIM_J * sizeof(elem_t));
  gemmini_mvin(B, B_sp_addr);

  // Configure execution: output-stationary, no activation, no shift
  gemmini_config_ex(OUTPUT_STATIONARY, 0, 0);

  // Preload zeros (no bias) and compute C = A * B
  gemmini_preload_zeros(C_sp_addr);
  gemmini_compute_preloaded(A_sp_addr, B_sp_addr);

  // Move result from scratchpad to main memory
  gemmini_config_st(MAT_DIM_J * sizeof(elem_t));
  gemmini_mvout(C, C_sp_addr);

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
