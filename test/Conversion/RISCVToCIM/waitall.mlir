// RUN: torch-mlir-opt <%s --convert-rair-to-cim  %s | FileCheck %s

// CHECK-LABEL: func.func @wait() {
// CHECK: %[[PTR0:.*]] = call @llvm.riscv.sync_1_0() : () -> !llvm.ptr
// CHECK: %[[PTR1:.*]] = call @llvm.riscv.sync_1_1(%[[PTR0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: call @llvm.riscv.sync_0_2(%[[PTR0]], %[[PTR1]]) : (!llvm.ptr, !llvm.ptr) -> ()
func.func @wait() {
  %1 = rair.wait_all async
  %2 = rair.wait_all async [%1]
  rair.wait_all [%1, %2]
  return
}