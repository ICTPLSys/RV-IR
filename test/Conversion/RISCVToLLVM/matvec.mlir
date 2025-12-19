// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
    %0 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "riscv.constant"() {value = dense<[1.0, 3.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>

    // "riscv.print"(%0) : (tensor<2x3xf32>) -> ()
    // "riscv.print"(%1) : (tensor<3xf32>) -> ()
    %3 = "riscv.matvec"(%0, %1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>  
    "riscv.print"(%3) : (tensor<2xf32>) -> ()

    return
}