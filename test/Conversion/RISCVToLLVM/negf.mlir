// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {

   %0 = "riscv.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
   %4 = "riscv.negf"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>

   "riscv.print"(%4) : (tensor<2x3xf32>) -> ()

  return
}