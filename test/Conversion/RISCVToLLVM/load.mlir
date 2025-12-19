// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm  %s | FileCheck %s

// CHECK: llvm.func @load()
func.func @load() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  
  %0 = riscv.alloc : memref<1024xi32, 0>
  %1 = riscv.alloc : memref<16xi32, 0>
  
  %e0 = riscv.load async ( %1[%c0][%c16][%c16], %0[%c0][%c16][%c16]) 
  : (memref<16xi32, 0>, memref<1024xi32, 0>)
  
    riscv.dealloc %0 : memref<1024xi32, 0>
  
  return
}

