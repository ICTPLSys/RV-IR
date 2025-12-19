// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()

func.func @main() {
    %0 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    %shape = "riscv.constant"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    // %1 = "riscv.reshape"(%0, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %1 = "riscv.reshape"(%0, %shape) : (tensor<2x3xf64>, tensor<2xi32>) -> tensor<3x2xf64>

    "riscv.print"(%1) : (tensor<3x2xf64>) -> ()
    return
}
