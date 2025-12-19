// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm  %s | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
  // Integer constants of different bit widths
  // tensor constants (floating-point)
   %0 = "riscv.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "riscv.constant"() {value = dense<[[7.000000e+00, 8.000000e+00, 9.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>

  // 2. tensor floating-point addition (same type)
  %4 = "riscv.add"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  "riscv.print"(%4) : (tensor<2x3xf32>) -> ()

  return
}
