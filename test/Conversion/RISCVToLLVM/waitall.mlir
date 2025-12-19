// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm  %s | FileCheck %s
// CHECK-LABEL: llvm.func @wait

func.func @wait() {
  %1 = riscv.wait_all async
  riscv.wait_all [%1]
  %2 = riscv.wait_all async [%1]
  riscv.wait_all [%1, %2]
  return
}