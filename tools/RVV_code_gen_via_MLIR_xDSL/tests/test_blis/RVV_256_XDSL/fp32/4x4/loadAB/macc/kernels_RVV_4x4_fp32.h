#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_1x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_1x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_2x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_2x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_3x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_3x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_3x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_3x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_3x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_3x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_3x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_3x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_4x1_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_4x1_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_4x2_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_4x2_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_4x3_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_4x3_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================
void gemm_RVV_4x4_b0_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);
// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:19
//   MR         : 4
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================
void gemm_RVV_4x4_b1_col_fp32(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

#ifdef __cplusplus
}
#endif
