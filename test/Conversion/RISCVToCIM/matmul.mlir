// RUN: torch-mlir-opt <%s --convert-rocc-to-cim | FileCheck %s

// CHECK: call @llvm.riscv.trans.drv(%{{.*}}) : (i32) -> i32    

// CHECK: call @llvm.riscv.vv.v.drv(%{{.*}}, %{{.*}}) : (i32, i32) -> i32  
module attributes {torch.debug_module_name = "Linear"} {
  memref.global "private" constant @__constant_64x32xi64 : memref<64x32xi64> = dense<0> {alignment = 64 : i64}
  func.func @main(%arg0: memref<16x32xi64, strided<[?, ?], offset: ?>>) -> memref<16x64xi64> {
    %cst = arith.constant 0 : i64
    %0 = memref.get_global @__constant_64x32xi64 : memref<64x32xi64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xi64>
    rocc.transpose ins(%0 : memref<64x32xi64>) outs(%alloc : memref<32x64xi64>) {permutation = array<i64: 1, 0>}
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x64xi64>
    linalg.fill ins(%cst : i64) outs(%alloc_0 : memref<16x64xi64>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x64xi64>
    memref.copy %alloc_0, %alloc_1 : memref<16x64xi64> to memref<16x64xi64>
    rocc.matmul ins(%arg0, %alloc : memref<16x32xi64, strided<[?, ?], offset: ?>>, memref<32x64xi64>) outs(%alloc_1 : memref<16x64xi64>)
    return %alloc_1 : memref<16x64xi64>
  }
}