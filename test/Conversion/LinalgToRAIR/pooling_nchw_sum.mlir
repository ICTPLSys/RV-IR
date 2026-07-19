// RUN: torch-mlir-opt <%s --convert-linalg-to-rair | FileCheck %s

// CHECK: rair.pooling_nchw_sum
func.func @main(%input: memref<1x4x9x9xf32>,
                %kernel: memref<9x9xf32>,
                %output: memref<1x4x1x1xf32>) {
  linalg.pooling_nchw_sum
    {dilations = dense<1> : vector<2xi64>,
     strides = dense<9> : vector<2xi64>}
    ins(%input, %kernel : memref<1x4x9x9xf32>, memref<9x9xf32>)
    outs(%output : memref<1x4x1x1xf32>)
  return
}
