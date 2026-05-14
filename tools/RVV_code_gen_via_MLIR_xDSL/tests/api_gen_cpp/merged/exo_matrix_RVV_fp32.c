#include <stdlib.h>
#include "exo_matrix_RVV_fp32.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc((4 + 1) * sizeof(ukrFunction***));
    for (int i = 0; i < 4 + 1; i++) {
        matrix[i] = (ukrFunction***)malloc((4 + 1) * sizeof(ukrFunction**));
        for (int j = 0; j < 4 + 1; j++) {
            matrix[i][j] = (ukrFunction**)malloc(2 * sizeof(ukrFunction*));
            for (int b = 0; b < 2; b++) {
                matrix[i][j][b] = (ukrFunction*)malloc(1 * sizeof(ukrFunction));
            }
        }
    }
    return matrix;
}

void fillMatrix(ukrFunction**** matrix) {
    *matrix[0][0][0] = 	(ukrFunction)NULL;
    *matrix[0][0][1] = 	(ukrFunction)NULL;
    *matrix[0][1][0] = 	(ukrFunction)NULL;
    *matrix[0][1][1] = 	(ukrFunction)NULL;
    *matrix[0][2][0] = 	(ukrFunction)NULL;
    *matrix[0][2][1] = 	(ukrFunction)NULL;
    *matrix[0][3][0] = 	(ukrFunction)NULL;
    *matrix[0][3][1] = 	(ukrFunction)NULL;
    *matrix[0][4][0] = 	(ukrFunction)NULL;
    *matrix[0][4][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RVV_1x1_b0_col_fp32;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RVV_1x1_b1_col_fp32;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RVV_1x2_b0_col_fp32;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RVV_1x2_b1_col_fp32;
    *matrix[1][3][0] = 	(ukrFunction)gemm_RVV_1x3_b0_col_fp32;
    *matrix[1][3][1] = 	(ukrFunction)gemm_RVV_1x3_b1_col_fp32;
    *matrix[1][4][0] = 	(ukrFunction)gemm_RVV_1x4_b0_col_fp32;
    *matrix[1][4][1] = 	(ukrFunction)gemm_RVV_1x4_b1_col_fp32;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RVV_2x1_b0_col_fp32;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RVV_2x1_b1_col_fp32;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RVV_2x2_b0_col_fp32;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RVV_2x2_b1_col_fp32;
    *matrix[2][3][0] = 	(ukrFunction)gemm_RVV_2x3_b0_col_fp32;
    *matrix[2][3][1] = 	(ukrFunction)gemm_RVV_2x3_b1_col_fp32;
    *matrix[2][4][0] = 	(ukrFunction)gemm_RVV_2x4_b0_col_fp32;
    *matrix[2][4][1] = 	(ukrFunction)gemm_RVV_2x4_b1_col_fp32;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RVV_3x1_b0_col_fp32;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RVV_3x1_b1_col_fp32;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RVV_3x2_b0_col_fp32;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RVV_3x2_b1_col_fp32;
    *matrix[3][3][0] = 	(ukrFunction)gemm_RVV_3x3_b0_col_fp32;
    *matrix[3][3][1] = 	(ukrFunction)gemm_RVV_3x3_b1_col_fp32;
    *matrix[3][4][0] = 	(ukrFunction)gemm_RVV_3x4_b0_col_fp32;
    *matrix[3][4][1] = 	(ukrFunction)gemm_RVV_3x4_b1_col_fp32;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RVV_4x1_b0_col_fp32;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RVV_4x1_b1_col_fp32;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RVV_4x2_b0_col_fp32;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RVV_4x2_b1_col_fp32;
    *matrix[4][3][0] = 	(ukrFunction)gemm_RVV_4x3_b0_col_fp32;
    *matrix[4][3][1] = 	(ukrFunction)gemm_RVV_4x3_b1_col_fp32;
    *matrix[4][4][0] = 	(ukrFunction)gemm_RVV_4x4_b0_col_fp32;
    *matrix[4][4][1] = 	(ukrFunction)gemm_RVV_4x4_b1_col_fp32;
}

void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 4 + 1; i++) {
        for (int j = 0; j < 4 + 1; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}
