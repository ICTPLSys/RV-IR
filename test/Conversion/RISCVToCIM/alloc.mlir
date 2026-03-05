// RUN: torch-mlir-opt <%s --convert-riscv-to-cim  %s | FileCheck %s

// CHECK: call @__npu_mem_malloc
// CHECK: call @__npu_mem_free
func.func @alloc() {
  // %c0 = arith.constant 0 : index
  // %c16 = arith.constant 16 : index
  
  %0 = riscv.alloc : memref<1024xi32, 0>
  riscv.dealloc %0 : memref<1024xi32, 0>
  return
}
