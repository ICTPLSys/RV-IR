def generate_exo_matrix_c(max_mr, max_nr):
    print("void fillMatrix(ukrFunction**** matrix) {")
    for mr in range(max_mr + 1):
        for nr in range(max_nr + 1):
            if mr == 0 or nr == 0:
                print(f"    *matrix[{mr}][{nr}][0] = \t(ukrFunction)NULL;")
                print(f"    *matrix[{mr}][{nr}][1] = \t(ukrFunction)NULL;")
            else:
                print(f"    *matrix[{mr}][{nr}][0] = \t(ukrFunction)gemm_RVV_{mr}x{nr}_b0_col_fp32;")
                print(f"    *matrix[{mr}][{nr}][1] = \t(ukrFunction)gemm_RVV_{mr}x{nr}_b1_col_fp32;")
    print("}")
