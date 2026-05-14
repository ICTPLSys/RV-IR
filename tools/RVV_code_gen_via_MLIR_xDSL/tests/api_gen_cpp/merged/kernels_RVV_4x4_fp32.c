#include "kernels_RVV_4x4_fp32.h"
#include <riscv_vector.h>

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
// Merged from xdsl_api_gemm_1x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t v22 = v11; v22 < v1; v22 += v12) {
    size_t v23 = v22 * v4;
    size_t v24 = 1;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = v22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t v23 = v11; v23 < v1; v23 += v12) {
    size_t v24 = v23 * v4;
    size_t v25 = 1;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = v23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t v24 = v11; v24 < v1; v24 += v12) {
    size_t v25 = v24 * v4;
    size_t v26 = 1;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = v24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v12);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v12);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x3_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 1;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v12);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v12);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x3_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t v30 = v11; v30 < v1; v30 += v12) {
    size_t v31 = v30 * v4;
    size_t v32 = 1;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = v30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v12);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v12);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x4_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_1x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v12);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t v28 = v11; v28 < v1; v28 += v12) {
    size_t v29 = v28 * v4;
    size_t v30 = 1;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = v28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v12);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v12);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v12);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v12);
  return;
}



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
// Merged from xdsl_api_gemm_1x4_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 1
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_1x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v12);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v12);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v12);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v12);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t v34 = v11; v34 < v1; v34 += v12) {
    size_t v35 = v34 * v4;
    size_t v36 = 1;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = v34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v12);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v12);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v12);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v12);
  return;
}



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
// Merged from xdsl_api_gemm_2x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t v22 = v11; v22 < v1; v22 += v12) {
    size_t v23 = v22 * v4;
    size_t v24 = 2;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = v22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t v23 = v11; v23 < v1; v23 += v12) {
    size_t v24 = v23 * v4;
    size_t v25 = 2;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = v23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t v24 = v11; v24 < v1; v24 += v12) {
    size_t v25 = v24 * v4;
    size_t v26 = 2;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = v24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v13);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v13);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x3_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 2;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v13);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v13);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x3_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t v30 = v11; v30 < v1; v30 += v12) {
    size_t v31 = v30 * v4;
    size_t v32 = 2;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = v30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v13);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v13);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x4_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_2x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v13);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t v28 = v11; v28 < v1; v28 += v12) {
    size_t v29 = v28 * v4;
    size_t v30 = 2;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = v28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v13);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v13);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v13);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v13);
  return;
}



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
// Merged from xdsl_api_gemm_2x4_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 2
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_2x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v13);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v13);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v13);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v13);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t v34 = v11; v34 < v1; v34 += v12) {
    size_t v35 = v34 * v4;
    size_t v36 = 2;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = v34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v13);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v13);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v13);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v13);
  return;
}



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
// Merged from xdsl_api_gemm_3x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t v22 = v11; v22 < v1; v22 += v12) {
    size_t v23 = v22 * v4;
    size_t v24 = 3;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = v22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t v23 = v11; v23 < v1; v23 += v12) {
    size_t v24 = v23 * v4;
    size_t v25 = 3;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = v23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t v24 = v11; v24 < v1; v24 += v12) {
    size_t v25 = v24 * v4;
    size_t v26 = 3;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = v24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v14);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 3;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v14);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x3_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 3;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v14);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v14);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x3_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 3
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t v30 = v11; v30 < v1; v30 += v12) {
    size_t v31 = v30 * v4;
    size_t v32 = 3;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = v30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v14);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v14);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x4_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_3x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v14);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t v28 = v11; v28 < v1; v28 += v12) {
    size_t v29 = v28 * v4;
    size_t v30 = 3;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = v28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v14);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v14);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v14);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v14);
  return;
}



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
// Merged from xdsl_api_gemm_3x4_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 3
//   NR         : 4
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_3x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v14);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v14);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v14);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v14);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t v34 = v11; v34 < v1; v34 += v12) {
    size_t v35 = v34 * v4;
    size_t v36 = 3;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = v34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v14);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v14);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v14);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v14);
  return;
}



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
// Merged from xdsl_api_gemm_4x1_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x1_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21;
  v21 = v20;
  for (size_t v22 = v11; v22 < v1; v22 += v12) {
    size_t v23 = v22 * v4;
    size_t v24 = 4;
    float* v25 = &v3[v23];
    vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v24);
    size_t v27 = v22 * v6;
    vfloat32m1_t v28 = v21;
    float v29 = v5[v27];
    vfloat32m1_t v30 = __riscv_vfmacc_vf_f32m1(v28, v29, v26, v24);
    v21 = v30;
  }
  vfloat32m1_t v31 = v21;
  float* v32 = &v8[v11];
  __riscv_vse32_v_f32m1(v32, v31, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x1_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 4
//   NR         : 1
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b1  (beta=1: C = alpha*A*B + beta*C)
// ============================================================

void gemm_RVV_4x1_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  vfloat32m1_t v22;
  v22 = v21;
  for (size_t v23 = v11; v23 < v1; v23 += v12) {
    size_t v24 = v23 * v4;
    size_t v25 = 4;
    float* v26 = &v3[v24];
    vfloat32m1_t v27 = __riscv_vle32_v_f32m1(v26, v25);
    size_t v28 = v23 * v6;
    vfloat32m1_t v29 = v22;
    float v30 = v5[v28];
    vfloat32m1_t v31 = __riscv_vfmacc_vf_f32m1(v29, v30, v27, v25);
    v22 = v31;
  }
  vfloat32m1_t v32 = v22;
  float* v33 = &v8[v11];
  __riscv_vse32_v_f32m1(v33, v32, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x2_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

// ============================================================
// xDSL Generated Kernel
// ------------------------------------------------------------
//   Generated   : 2026-05-14 06:47:18
//   MR         : 4
//   NR         : 2
//   VLEN       : 8  (fp32 elements per vector register)
//   VLEN_BITS  : 256  (bit-width, VLEN * 32)
//   Beta mode  : b0  (beta=0: C = alpha*A*B)
// ============================================================

void gemm_RVV_4x2_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22;
  v22 = v20;
  vfloat32m1_t v23;
  v23 = v21;
  for (size_t v24 = v11; v24 < v1; v24 += v12) {
    size_t v25 = v24 * v4;
    size_t v26 = 4;
    float* v27 = &v3[v25];
    vfloat32m1_t v28 = __riscv_vle32_v_f32m1(v27, v26);
    size_t v29 = v24 * v6;
    size_t v30 = v29 + v12;
    vfloat32m1_t v31 = v22;
    float v32 = v5[v29];
    vfloat32m1_t v33 = __riscv_vfmacc_vf_f32m1(v31, v32, v28, v26);
    vfloat32m1_t v34 = v23;
    float v35 = v5[v30];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v28, v26);
    v22 = v33;
    v23 = v36;
  }
  vfloat32m1_t v37 = v22;
  vfloat32m1_t v38 = v23;
  float* v39 = &v8[v11];
  __riscv_vse32_v_f32m1(v39, v37, v15);
  float* v40 = &v8[v9];
  __riscv_vse32_v_f32m1(v40, v38, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x2_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

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

void gemm_RVV_4x2_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v23;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 4;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    vfloat32m1_t v33 = v24;
    float v34 = v5[v31];
    vfloat32m1_t v35 = __riscv_vfmacc_vf_f32m1(v33, v34, v30, v28);
    vfloat32m1_t v36 = v25;
    float v37 = v5[v32];
    vfloat32m1_t v38 = __riscv_vfmacc_vf_f32m1(v36, v37, v30, v28);
    v24 = v35;
    v25 = v38;
  }
  vfloat32m1_t v39 = v24;
  vfloat32m1_t v40 = v25;
  float* v41 = &v8[v11];
  __riscv_vse32_v_f32m1(v41, v39, v15);
  float* v42 = &v8[v9];
  __riscv_vse32_v_f32m1(v42, v40, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x3_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

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

void gemm_RVV_4x3_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23;
  v23 = v20;
  vfloat32m1_t v24;
  v24 = v21;
  vfloat32m1_t v25;
  v25 = v22;
  for (size_t v26 = v11; v26 < v1; v26 += v12) {
    size_t v27 = v26 * v4;
    size_t v28 = 4;
    float* v29 = &v3[v27];
    vfloat32m1_t v30 = __riscv_vle32_v_f32m1(v29, v28);
    size_t v31 = v26 * v6;
    size_t v32 = v31 + v12;
    size_t v33 = v31 + v13;
    vfloat32m1_t v34 = v23;
    float v35 = v5[v31];
    vfloat32m1_t v36 = __riscv_vfmacc_vf_f32m1(v34, v35, v30, v28);
    vfloat32m1_t v37 = v24;
    float v38 = v5[v32];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v30, v28);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v33];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v30, v28);
    v23 = v36;
    v24 = v39;
    v25 = v42;
  }
  vfloat32m1_t v43 = v23;
  vfloat32m1_t v44 = v24;
  vfloat32m1_t v45 = v25;
  float* v46 = &v8[v11];
  __riscv_vse32_v_f32m1(v46, v43, v15);
  float* v47 = &v8[v9];
  __riscv_vse32_v_f32m1(v47, v44, v15);
  size_t v48 = v13 * v9;
  float* v49 = &v8[v48];
  __riscv_vse32_v_f32m1(v49, v45, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x3_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

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

void gemm_RVV_4x3_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  vfloat32m1_t v27;
  v27 = v21;
  vfloat32m1_t v28;
  v28 = v23;
  vfloat32m1_t v29;
  v29 = v26;
  for (size_t v30 = v11; v30 < v1; v30 += v12) {
    size_t v31 = v30 * v4;
    size_t v32 = 4;
    float* v33 = &v3[v31];
    vfloat32m1_t v34 = __riscv_vle32_v_f32m1(v33, v32);
    size_t v35 = v30 * v6;
    size_t v36 = v35 + v12;
    size_t v37 = v35 + v13;
    vfloat32m1_t v38 = v27;
    float v39 = v5[v35];
    vfloat32m1_t v40 = __riscv_vfmacc_vf_f32m1(v38, v39, v34, v32);
    vfloat32m1_t v41 = v28;
    float v42 = v5[v36];
    vfloat32m1_t v43 = __riscv_vfmacc_vf_f32m1(v41, v42, v34, v32);
    vfloat32m1_t v44 = v29;
    float v45 = v5[v37];
    vfloat32m1_t v46 = __riscv_vfmacc_vf_f32m1(v44, v45, v34, v32);
    v27 = v40;
    v28 = v43;
    v29 = v46;
  }
  vfloat32m1_t v47 = v27;
  vfloat32m1_t v48 = v28;
  vfloat32m1_t v49 = v29;
  float* v50 = &v8[v11];
  __riscv_vse32_v_f32m1(v50, v47, v15);
  float* v51 = &v8[v9];
  __riscv_vse32_v_f32m1(v51, v48, v15);
  size_t v52 = v13 * v9;
  float* v53 = &v8[v52];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x4_b0.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

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

void gemm_RVV_4x4_b0_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  vfloat32m1_t v20 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v21 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v22 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v23 = __riscv_vfmv_v_f_f32m1(v10, v15);
  vfloat32m1_t v24;
  v24 = v20;
  vfloat32m1_t v25;
  v25 = v21;
  vfloat32m1_t v26;
  v26 = v22;
  vfloat32m1_t v27;
  v27 = v23;
  for (size_t v28 = v11; v28 < v1; v28 += v12) {
    size_t v29 = v28 * v4;
    size_t v30 = 4;
    float* v31 = &v3[v29];
    vfloat32m1_t v32 = __riscv_vle32_v_f32m1(v31, v30);
    size_t v33 = v28 * v6;
    size_t v34 = v33 + v12;
    size_t v35 = v33 + v13;
    size_t v36 = v33 + v14;
    vfloat32m1_t v37 = v24;
    float v38 = v5[v33];
    vfloat32m1_t v39 = __riscv_vfmacc_vf_f32m1(v37, v38, v32, v30);
    vfloat32m1_t v40 = v25;
    float v41 = v5[v34];
    vfloat32m1_t v42 = __riscv_vfmacc_vf_f32m1(v40, v41, v32, v30);
    vfloat32m1_t v43 = v26;
    float v44 = v5[v35];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v32, v30);
    vfloat32m1_t v46 = v27;
    float v47 = v5[v36];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v32, v30);
    v24 = v39;
    v25 = v42;
    v26 = v45;
    v27 = v48;
  }
  vfloat32m1_t v49 = v24;
  vfloat32m1_t v50 = v25;
  vfloat32m1_t v51 = v26;
  vfloat32m1_t v52 = v27;
  float* v53 = &v8[v11];
  __riscv_vse32_v_f32m1(v53, v49, v15);
  float* v54 = &v8[v9];
  __riscv_vse32_v_f32m1(v54, v50, v15);
  size_t v55 = v13 * v9;
  float* v56 = &v8[v55];
  __riscv_vse32_v_f32m1(v56, v51, v15);
  size_t v57 = v14 * v9;
  float* v58 = &v8[v57];
  __riscv_vse32_v_f32m1(v58, v52, v15);
  return;
}



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
// Merged from xdsl_api_gemm_4x4_b1.cpp
#include <stdio.h>
#include <riscv_vector.h>
#include <math.h>

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

void gemm_RVV_4x4_b1_col_fp32(void* ctxt, int v1, float* v2, float* v3, int v4, float* v5, int v6, float* v7, float* v8, int v9) {
  float v10 = 0.0e+00f;
  size_t v11 = 0;
  size_t v12 = 1;
  size_t v13 = 2;
  size_t v14 = 3;
  size_t v15 = 4;
  size_t v16 = 5;
  size_t v17 = 6;
  size_t v18 = 7;
  size_t v19 = 8;
  float* v20 = &v8[v11];
  vfloat32m1_t v21 = __riscv_vle32_v_f32m1(v20, v15);
  float* v22 = &v8[v9];
  vfloat32m1_t v23 = __riscv_vle32_v_f32m1(v22, v15);
  size_t v24 = v13 * v9;
  float* v25 = &v8[v24];
  vfloat32m1_t v26 = __riscv_vle32_v_f32m1(v25, v15);
  size_t v27 = v14 * v9;
  float* v28 = &v8[v27];
  vfloat32m1_t v29 = __riscv_vle32_v_f32m1(v28, v15);
  vfloat32m1_t v30;
  v30 = v21;
  vfloat32m1_t v31;
  v31 = v23;
  vfloat32m1_t v32;
  v32 = v26;
  vfloat32m1_t v33;
  v33 = v29;
  for (size_t v34 = v11; v34 < v1; v34 += v12) {
    size_t v35 = v34 * v4;
    size_t v36 = 4;
    float* v37 = &v3[v35];
    vfloat32m1_t v38 = __riscv_vle32_v_f32m1(v37, v36);
    size_t v39 = v34 * v6;
    size_t v40 = v39 + v12;
    size_t v41 = v39 + v13;
    size_t v42 = v39 + v14;
    vfloat32m1_t v43 = v30;
    float v44 = v5[v39];
    vfloat32m1_t v45 = __riscv_vfmacc_vf_f32m1(v43, v44, v38, v36);
    vfloat32m1_t v46 = v31;
    float v47 = v5[v40];
    vfloat32m1_t v48 = __riscv_vfmacc_vf_f32m1(v46, v47, v38, v36);
    vfloat32m1_t v49 = v32;
    float v50 = v5[v41];
    vfloat32m1_t v51 = __riscv_vfmacc_vf_f32m1(v49, v50, v38, v36);
    vfloat32m1_t v52 = v33;
    float v53 = v5[v42];
    vfloat32m1_t v54 = __riscv_vfmacc_vf_f32m1(v52, v53, v38, v36);
    v30 = v45;
    v31 = v48;
    v32 = v51;
    v33 = v54;
  }
  vfloat32m1_t v55 = v30;
  vfloat32m1_t v56 = v31;
  vfloat32m1_t v57 = v32;
  vfloat32m1_t v58 = v33;
  float* v59 = &v8[v11];
  __riscv_vse32_v_f32m1(v59, v55, v15);
  float* v60 = &v8[v9];
  __riscv_vse32_v_f32m1(v60, v56, v15);
  size_t v61 = v13 * v9;
  float* v62 = &v8[v61];
  __riscv_vse32_v_f32m1(v62, v57, v15);
  size_t v63 = v14 * v9;
  float* v64 = &v8[v63];
  __riscv_vse32_v_f32m1(v64, v58, v15);
  return;
}



