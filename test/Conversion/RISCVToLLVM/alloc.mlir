// RUN: torch-mlir-opt <%s --convert-rair-to-affine  %s | FileCheck %s

// CHECK: memref.alloc
// CHECK: memref.dealloc
func.func @alloc() {
  // %c0 = arith.constant 0 : index
  // %c16 = arith.constant 16 : index
  
  %0 = rair.alloc : memref<1024xi32, 6>
  rair.dealloc %0 : memref<1024xi32, 6>
  return
}
