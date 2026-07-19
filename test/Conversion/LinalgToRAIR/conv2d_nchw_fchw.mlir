// RUN: torch-mlir-opt <%s --convert-linalg-to-rair | FileCheck %s

// CHECK: rair.conv_2d_nchw_fchw
func.func @main(%input: memref<1x3x11x11xf32>,
                %kernel: memref<4x3x3x3xf32>,
                %output: memref<1x4x9x9xf32>) {
  linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : vector<2xi64>,
     strides = dense<1> : vector<2xi64>}
    ins(%input, %kernel : memref<1x3x11x11xf32>, memref<4x3x3x3xf32>)
    outs(%output : memref<1x4x9x9xf32>)
  return
}
