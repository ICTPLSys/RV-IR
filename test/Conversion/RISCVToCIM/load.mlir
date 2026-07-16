// RUN: torch-mlir-opt <%s --convert-rair-to-cim %s | FileCheck %s

// CHECK: call @llvm.riscv.load(%{{.*}}, %{{.*}}) : (i32, i32) -> i32  
func.func @load() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  
  %0 = rair.alloc : memref<1024xi32, 0>
  %1 = rair.alloc : memref<16xi32, 0>
  
  %e0 = rair.load async ( %1[%c0][%c16][%c16], %0[%c0][%c16][%c16]) 
  : (memref<16xi32, 0>, memref<1024xi32, 0>)
  
    rair.dealloc %0 : memref<1024xi32, 0>
  
  return
}

