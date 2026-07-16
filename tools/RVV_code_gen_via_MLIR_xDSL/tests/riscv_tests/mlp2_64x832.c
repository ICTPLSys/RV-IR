// Auto-generated Gemmini multi-layer MLP from RAIR MLIR
// 2 layers:
//   layer 0: [64x832] * [832x832] -> [64x832]
//   layer 1: [64x832] * [832x64] -> [64x64]

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_testutils.h"

#define L0_DIM_I 64
#define L0_DIM_K 832
#define L0_DIM_J 832
#define L1_DIM_I 64
#define L1_DIM_K 832
#define L1_DIM_J 64

static elem_t input_mat[L0_DIM_I][L0_DIM_K] row_align(1);
static elem_t weights0[L0_DIM_K][L0_DIM_J] row_align(1);
static elem_t inter0[L0_DIM_I][L0_DIM_J] row_align(1);
static elem_t weights1[L1_DIM_K][L1_DIM_J] row_align(1);
static elem_t output_mat[L1_DIM_I][L1_DIM_J] row_align(1);

static elem_t inter0_gold[L0_DIM_I][L0_DIM_J];
static elem_t output_gold[L1_DIM_I][L1_DIM_J];

static elem_t saturate(full_t x) {
  #ifndef ELEM_T_IS_FLOAT
  if (x > elem_t_max) return elem_t_max;
  if (x < elem_t_min) return elem_t_min;
  #endif
  return (elem_t)x;
}

static void init_data(void) {
  for (size_t i = 0; i < L0_DIM_I; i++)
    for (size_t j = 0; j < L0_DIM_K; j++)
      input_mat[i][j] = (elem_t)((i + 3 * j) % 9 - 4);

  for (size_t i = 0; i < L0_DIM_K; i++)
    for (size_t j = 0; j < L0_DIM_J; j++)
      weights0[i][j] = (elem_t)((2 * i + j) % 7 - 3);

  for (size_t i = 0; i < L1_DIM_K; i++)
    for (size_t j = 0; j < L1_DIM_J; j++)
      weights1[i][j] = (elem_t)((4 * i + j) % 7 - 3);

}

static void cpu_reference(void) {
  for (size_t i = 0; i < L0_DIM_I; i++) {
    for (size_t j = 0; j < L0_DIM_J; j++) {
      full_t sum = 0;
      for (size_t k = 0; k < L0_DIM_K; k++)
        sum += (full_t)input_mat[i][k] * (full_t)weights0[k][j];
      inter0_gold[i][j] = saturate(sum);
    }
  }
  // Copy gold to inter0 for next layer CPU ref
  for (size_t i = 0; i < L0_DIM_I; i++)
    for (size_t j = 0; j < L0_DIM_J; j++)
      inter0[i][j] = inter0_gold[i][j];

  for (size_t i = 0; i < L1_DIM_I; i++) {
    for (size_t j = 0; j < L1_DIM_J; j++) {
      full_t sum = 0;
      for (size_t k = 0; k < L1_DIM_K; k++)
        sum += (full_t)inter0[i][k] * (full_t)weights1[k][j];
      output_gold[i][j] = saturate(sum);
    }
  }

}

int main(void) {
  #ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
  #endif

  printf("MLP 2-layer Gemmini program\n");
  printf("  layer 0: [%d x %d] * [%d x %d] -> [%d x %d]\n",
      L0_DIM_I, L0_DIM_K, L0_DIM_K, L0_DIM_J,
      L0_DIM_I, L0_DIM_J);
  printf("  layer 1: [%d x %d] * [%d x %d] -> [%d x %d]\n",
      L1_DIM_I, L1_DIM_K, L1_DIM_K, L1_DIM_J,
      L1_DIM_I, L1_DIM_J);

  init_data();

  printf("Computing CPU reference...\n");
  uint64_t cpu_start = read_cycles();
  cpu_reference();
  uint64_t cpu_end = read_cycles();
  printf("CPU reference took %llu cycles\n", cpu_end - cpu_start);

  for (size_t i = 0; i < L0_DIM_I; i++)
    for (size_t j = 0; j < L0_DIM_J; j++)
      inter0[i][j] = 0;

  gemmini_flush(0);

  printf("Running on Gemmini...\n");
  uint64_t gemmini_start = read_cycles();

  // ---- Layer 0: [64x832] * [832x832] ----
  uint64_t l0_start = read_cycles();
  tiled_matmul_auto(L0_DIM_I, L0_DIM_J, L0_DIM_K,
      (elem_t*)input_mat, (elem_t*)weights0, NULL, (elem_t*)inter0,
      L0_DIM_K, L0_DIM_J, L0_DIM_J, L0_DIM_J,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
      false, false,
      false, false,
      0,
      WS);
  gemmini_fence();
  uint64_t l0_end = read_cycles();
  printf("  Layer 0 Gemmini: %llu cycles\n", l0_end - l0_start);

  // ---- Layer 1: [64x832] * [832x64] ----
  uint64_t l1_start = read_cycles();
  tiled_matmul_auto(L1_DIM_I, L1_DIM_J, L1_DIM_K,
      (elem_t*)inter0, (elem_t*)weights1, NULL, (elem_t*)output_mat,
      L1_DIM_K, L1_DIM_J, L1_DIM_J, L1_DIM_J,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
      false, false,
      false, false,
      0,
      WS);
  gemmini_fence();
  uint64_t l1_end = read_cycles();
  printf("  Layer 1 Gemmini: %llu cycles\n", l1_end - l1_start);

  uint64_t gemmini_end = read_cycles();
  printf("Total Gemmini: %llu cycles\n", gemmini_end - gemmini_start);

  // Verify final output
  int pass = 1;
  for (size_t i = 0; i < L1_DIM_I; i++) {
    for (size_t j = 0; j < L1_DIM_J; j++) {
      if (output_mat[i][j] != output_gold[i][j]) {
        printf("MISMATCH at output[%u][%u]: got %d, expected %d\n",
            (unsigned)i, (unsigned)j, output_mat[i][j], output_gold[i][j]);
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
