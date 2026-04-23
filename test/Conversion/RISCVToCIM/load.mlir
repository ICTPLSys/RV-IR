// RUN: torch-mlir-opt <%s --convert-rocc-to-cim %s | FileCheck %s

// CHECK: call @llvm.riscv.load(%{{.*}}, %{{.*}}) : (i32, i32) -> i32  
func.func @load() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  
  %0 = rocc.alloc : memref<1024xi32, 0>
  %1 = rocc.alloc : memref<16xi32, 0>
  
  %e0 = rocc.load async ( %1[%c0][%c16][%c16], %0[%c0][%c16][%c16]) 
  : (memref<16xi32, 0>, memref<1024xi32, 0>)
  
    rocc.dealloc %0 : memref<1024xi32, 0>
  
  return
}

