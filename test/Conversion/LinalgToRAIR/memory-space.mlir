// RUN: torch-mlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @typed_memory_spaces
// CHECK: %[[BUF:.*]] = rair.alloc_buffer %arg0 {memory_space = #rair.space<spad0>} : memref<4x4xf32, #rair.space<spad0>>
// CHECK: rair.transfer %arg1 to %[[BUF]] {dst_memory_space = #rair.space<spad0>, src_memory_space = #rair.space<gmem>}
// CHECK: rair.transfer %[[BUF]] to %arg2 {dst_memory_space = #rair.space<gmem>, src_memory_space = #rair.space<spad0>}
func.func @typed_memory_spaces(
    %ctx: !rair.context,
    %src: memref<4x4xf32, #rair.space<gmem>>,
    %dst: memref<4x4xf32, #rair.space<gmem>>) {
  %buf = rair.alloc_buffer %ctx {memory_space = #rair.space<spad0>}
      : memref<4x4xf32, #rair.space<spad0>>
  rair.transfer %src to %buf {
    src_memory_space = #rair.space<gmem>,
    dst_memory_space = #rair.space<spad0>
  } : memref<4x4xf32, #rair.space<gmem>>, memref<4x4xf32, #rair.space<spad0>>
  rair.transfer %buf to %dst {
    src_memory_space = #rair.space<spad0>,
    dst_memory_space = #rair.space<gmem>
  } : memref<4x4xf32, #rair.space<spad0>>, memref<4x4xf32, #rair.space<gmem>>
  return
}
