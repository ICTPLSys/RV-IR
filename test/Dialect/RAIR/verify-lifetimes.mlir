// RUN: torch-mlir-opt %s --rair-verify-lifetimes | FileCheck %s

// CHECK-LABEL: func.func @owned_resources
// CHECK: %[[CTX:.*]] = rair.acquire
// CHECK: %[[LHS:.*]] = rair.alloc_buffer %[[CTX]]
// CHECK: %[[RHS:.*]] = rair.alloc_buffer %[[CTX]]
// CHECK: rair.transfer %arg0 to %[[LHS]]
// CHECK: rair.dealloc_buffer %[[CTX]], %[[RHS]]
// CHECK: rair.dealloc_buffer %[[CTX]], %[[LHS]]
// CHECK: rair.await %[[CTX]]
// CHECK: rair.release %[[CTX]]
func.func @owned_resources(
    %src: memref<4x4xf32, #rair.space<gmem>>) {
  %ctx = rair.acquire {accelerator = "default"} : !rair.context
  %lhs = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4x4xf32, #rair.space<lmem>>
  %rhs = rair.alloc_buffer %ctx {memory_space = #rair.space<lmem>}
    : memref<4x4xf32, #rair.space<lmem>>
  rair.transfer %src to %lhs {
    src_memory_space = #rair.space<gmem>,
    dst_memory_space = #rair.space<lmem>
  } : memref<4x4xf32, #rair.space<gmem>>,
      memref<4x4xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %rhs
    : memref<4x4xf32, #rair.space<lmem>>
  rair.dealloc_buffer %ctx, %lhs
    : memref<4x4xf32, #rair.space<lmem>>
  rair.await %ctx : !rair.context
  rair.release %ctx : !rair.context
  return
}

// CHECK-LABEL: func.func @borrowed_context
func.func @borrowed_context(%ctx: !rair.context) {
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<spad0>}
    : memref<4xf32, #rair.space<spad0>>
  rair.dealloc_buffer %ctx, %buf : memref<4xf32, #rair.space<spad0>>
  return
}
