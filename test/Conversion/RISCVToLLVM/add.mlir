// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm  %s | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
  // Integer constants of different bit widths
  // tensor constants (floating-point)
   %0 = "rair.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "rair.constant"() {value = dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>

  // 2. tensor floating-point addition (same type)
  %4 = "rair.add"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  "rair.print"(%4) : (tensor<2x3xf32>) -> ()

  return
}
