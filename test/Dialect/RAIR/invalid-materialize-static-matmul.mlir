// RUN: torch-mlir-opt %s --rair-materialize-static-matmul -split-input-file -verify-diagnostics

func.func @dynamic_shape(
    %a: memref<?x32xf32, #rair.space<host>>,
    %b: memref<32x8xf32, #rair.space<host>>,
    %c: memref<?x8xf32, #rair.space<host>>) {
  // expected-error @+1 {{'linalg.matmul' op requires rank-2 static memref operand 0}}
  linalg.matmul
    ins(%a, %b
      : memref<?x32xf32, #rair.space<host>>,
        memref<32x8xf32, #rair.space<host>>)
    outs(%c : memref<?x8xf32, #rair.space<host>>)
  return
}

// -----

func.func @missing_core_space(
    %a: memref<4x4xf32>,
    %b: memref<4x4xf32, #rair.space<host>>,
    %c: memref<4x4xf32, #rair.space<host>>) {
  // expected-error @+1 {{'linalg.matmul' op requires operand 0 in #rair.space<host> or #rair.space<device>}}
  linalg.matmul
    ins(%a, %b
      : memref<4x4xf32>, memref<4x4xf32, #rair.space<host>>)
    outs(%c : memref<4x4xf32, #rair.space<host>>)
  return
}

// -----

func.func @tensor_semantics(
    %a: tensor<4x4xf32>, %b: tensor<4x4xf32>,
    %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{'linalg.matmul' op requires buffer semantics; tensor-result matmul is not supported}}
  %result = linalg.matmul
    ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}
