// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    %0 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "riscv.constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
    // "riscv.print"(%0) : (tensor<2x3xf32>) -> ()
    // "riscv.print"(%1) : (tensor<3x2xf32>) -> ()
    %2 = "riscv.matmul"(%0, %1) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    "riscv.print"(%2) : (tensor<2x2xf32>) -> ()
    return
}