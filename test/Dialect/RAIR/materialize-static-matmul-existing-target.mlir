// RUN: torch-mlir-opt %s --rair-materialize-static-matmul | FileCheck %s

// CHECK: rair.target @rair_default
// CHECK-SAME: kind = "gemmini"
// CHECK-NOT: rair.target @rair_default
// CHECK-LABEL: func.func @reuse_target
// CHECK: rair.scope @rair_default
// CHECK: rair.compute

rair.target @rair_default {
  kind = "gemmini",
  spad_bytes = 65536 : i64,
  acc_bytes = 16384 : i64
}

func.func @reuse_target(
    %a: memref<4x4xf32, #rair.space<host>>,
    %b: memref<4x4xf32, #rair.space<host>>,
    %c: memref<4x4xf32, #rair.space<host>>) {
  linalg.matmul
    ins(%a, %b
      : memref<4x4xf32, #rair.space<host>>,
        memref<4x4xf32, #rair.space<host>>)
    outs(%c : memref<4x4xf32, #rair.space<host>>)
  return
}
