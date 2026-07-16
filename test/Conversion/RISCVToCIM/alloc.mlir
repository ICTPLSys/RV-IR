// RUN: torch-mlir-opt <%s --convert-rair-to-cim  %s | FileCheck %s

// CHECK: call @__npu_mem_malloc
// CHECK: call @__npu_mem_free
func.func @alloc() {
  // %c0 = arith.constant 0 : index
  // %c16 = arith.constant 16 : index
  
  %0 = rair.alloc : memref<1024xi32, 0>
  rair.dealloc %0 : memref<1024xi32, 0>
  return
}
