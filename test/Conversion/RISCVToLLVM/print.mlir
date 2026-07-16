// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    %0 = "rair.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "rair.constant"() { value = dense<5.5> : tensor<f64> } : () -> tensor<f64>
    %2 = "rair.constant"() { value = dense<5.5> : tensor<f16> } : () -> tensor<f16>
    %3 = "rair.constant"() { value = dense<5.5> : tensor<bf16> } : () -> tensor<bf16>


    "rair.print"(%0) : (tensor<2x3xf64>) -> ()
    "rair.print"(%1) : (tensor<f64>) -> ()
    "rair.print"(%2) : (tensor<f16>) -> ()
    "rair.print"(%3) : (tensor<bf16>) -> ()



    return
}
