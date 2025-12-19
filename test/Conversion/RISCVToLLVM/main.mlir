// RUN: mlir-opt --convert-func-to-llvm --convert-arith-to-llvm %s | FileCheck %s
module {
  func.func @main() -> i32 {
    // CHECK: llvm.mlir.constant(42 : i32) : i32


    %v = arith.constant 42 : i32
    // CHECK: llvm.mlir.constant(dense<[2, 3]> : vector<2xi32>) : vector<2xi32>
    %0 = arith.constant dense<[2, 3]> : vector<2xi32>
    // CHECK: llvm.return %0 : i32
    return %v : i32
  }
}
