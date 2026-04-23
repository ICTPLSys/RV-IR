// RUN: torch-mlir-opt <%s --convert-rocc-to-affine  %s | FileCheck %s

// CHECK: memref.alloc
// CHECK: memref.dealloc
func.func @alloc() {
  // %c0 = arith.constant 0 : index
  // %c16 = arith.constant 16 : index
  
  %0 = rocc.alloc : memref<1024xi32, 6>
  rocc.dealloc %0 : memref<1024xi32, 6>
  return
}
