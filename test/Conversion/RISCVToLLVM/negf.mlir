// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {

   %0 = "rair.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
   %4 = "rair.negf"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>

   "rair.print"(%4) : (tensor<2x3xf32>) -> ()

  return
}