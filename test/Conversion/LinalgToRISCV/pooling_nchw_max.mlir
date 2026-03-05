// RUN: torch-mlir-opt <%s --convert-linalg-to-riscv | FileCheck %s

// CHECK: riscv.pooling_nchw_max
module attributes {torch.debug_module_name = "PoolingNCHWMax"} {
  memref.global "private" constant @__constant_3x3xf32
    : memref<3x3xf32> = dense<0.0> {alignment = 64 : i64}

  func.func @main(
      %input: memref<1x64x114x114xf32>
  ) -> memref<1x64x56x56xf32> {

    %kernel = memref.get_global @__constant_3x3xf32: memref<3x3xf32>
    %output = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>

    linalg.pooling_nchw_max{dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[2, 2]> : vector<2xi64>}
      ins(%input, %kernel: memref<1x64x114x114xf32>, memref<3x3xf32>)
      outs(%output: memref<1x64x56x56xf32>)
    return %output : memref<1x64x56x56xf32>
  }
}
