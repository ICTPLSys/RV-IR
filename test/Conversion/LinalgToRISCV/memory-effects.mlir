// RUN: torch-mlir-opt %s --canonicalize --cse | FileCheck %s

// This test protects the legacy RAIR operations while the dialect migrates to
// RAIR Core. In-place compute, copies, and observable I/O must not be removed
// as dead pure operations.

// CHECK-LABEL: func.func @preserve_effectful_ops
// CHECK: %[[TMP:.*]] = rair.alloc : memref<4x4xf32>
// CHECK: rair.transfer %arg2 to %[[TMP]]
// CHECK: rair.matmul
// CHECK: rair.transpose
// CHECK: rair.print %[[TMP]]
// CHECK: rair.world
// CHECK: rair.dealloc %[[TMP]]
func.func @preserve_effectful_ops(
    %lhs: memref<4x4xf32>,
    %rhs: memref<4x4xf32>,
    %out: memref<4x4xf32>) {
  %tmp = rair.alloc : memref<4x4xf32>
  rair.transfer %out to %tmp
    : memref<4x4xf32>, memref<4x4xf32>
  rair.matmul
    ins(%lhs, %rhs : memref<4x4xf32>, memref<4x4xf32>)
    outs(%tmp : memref<4x4xf32>)
  rair.transpose
    ins(%tmp : memref<4x4xf32>)
    outs(%out : memref<4x4xf32>)
    {permutation = array<i64: 1, 0>}
  rair.print %tmp : memref<4x4xf32>
  "rair.world"() : () -> ()
  rair.dealloc %tmp : memref<4x4xf32>
  return
}
