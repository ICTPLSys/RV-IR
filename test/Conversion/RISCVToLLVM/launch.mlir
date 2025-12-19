// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm  %s | FileCheck %s

// CHECK-LABEL: llvm.func @launch_1


func.func @launch_1() {
  %e0 = riscv.wait_all async
  %e1 = riscv.wait_all async [%e0]
  %t = riscv.launch async [%e0, %e1] () in () {
  }
  return
}
// CHECK-LABEL: llvm.func @launch_0
func.func @launch_0(%arg0: memref<16xf16>, %arg1: memref<16xf16>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  riscv.launch (%arg2, %arg3, %arg4) in (%arg5=%c4, %arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1) : memref<16xf16>, memref<16xf16> {
  }
  return
}