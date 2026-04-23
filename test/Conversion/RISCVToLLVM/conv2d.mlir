// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
  %input = "rocc.constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],[13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>

  %kernel = "rocc.constant"() {value = dense<[[0.1, 0.2],[0.3, 0.4]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  // %bias = "rocc.constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
  // %output = "rocc.conv2d"(%input, %kernel, %bias) : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<f32>) -> tensor<3x3xf32>
  %output = "rocc.conv2d"(%input, %kernel) : (tensor<4x4xf32>, tensor<2x2xf32>) -> tensor<3x3xf32>

  "rocc.print"(%output) : (tensor<3x3xf32>) -> ()
  return
}