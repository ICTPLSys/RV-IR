#pragma once
#include <stddef.h>
#include <stdint.h>
#include "kernels_RVV_4x4_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ukrFunction)(void* ctxt, int kc, float* alpha, float* A, int lda, float* B, int ldb, float* beta, float* C, int ldc);

ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);

#ifdef __cplusplus
}
#endif
