// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
  %input = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],[13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>

  %kernel = "riscv.constant"() {value = dense<[[0.1, 0.2],[0.3, 0.4]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  // %bias = "riscv.constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
  // %output = "riscv.conv2d"(%input, %kernel, %bias) : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<3x3xf32>
  %output = "riscv.conv2d"(%input, %kernel) : (tensor<4x4xf32>, tensor<2x2xf32>) -> tensor<3x3xf32>

  "riscv.print"(%output) : (tensor<3x3xf32>) -> ()
  return
}