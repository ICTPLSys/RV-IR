// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
    %0 = "rair.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "rair.constant"() {value = dense<[1.0, 3.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>

    // "rair.print"(%0) : (tensor<2x3xf32>) -> ()
    // "rair.print"(%1) : (tensor<3xf32>) -> ()
    %3 = "rair.matvec"(%0, %1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>  
    "rair.print"(%3) : (tensor<2xf32>) -> ()

    return
}