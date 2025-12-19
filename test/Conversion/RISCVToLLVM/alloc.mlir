// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm  %s | FileCheck %s

// CHECK: llvm.func @alloc()
func.func @alloc() {
  // %c0 = arith.constant 0 : index
  // %c16 = arith.constant 16 : index
  
  %0 = riscv.alloc : memref<1024xi32, 0>
  riscv.dealloc %0 : memref<1024xi32, 0>
  return
}
