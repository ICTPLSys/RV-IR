// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    %0 = "riscv.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "riscv.constant"() { value = dense<5.5> : tensor<f64> } : () -> tensor<f64>
    %2 = "riscv.constant"() { value = dense<5.5> : tensor<f16> } : () -> tensor<f16>
    %3 = "riscv.constant"() { value = dense<5.5> : tensor<bf16> } : () -> tensor<bf16>


    "riscv.print"(%0) : (tensor<2x3xf64>) -> ()
    "riscv.print"(%1) : (tensor<f64>) -> ()
    "riscv.print"(%2) : (tensor<f16>) -> ()
    "riscv.print"(%3) : (tensor<bf16>) -> ()



    return
}
