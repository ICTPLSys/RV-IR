// RUN: torch-mlir-opt <%s --convert-rocc-to-affine  %s | FileCheck %s

// CHECK-LABEL: func.func @wait
// CHECK: %[[T0:.*]] = async.execute {
// CHECK: %[[T1:.*]] = async.execute [%[[T0]]] {
// CHECK: async.await %[[T0]] : !async.token
// CHECK: async.await %[[T1]] : !async.token
func.func @wait() {
  %1 = rocc.wait_all async
  %2 = rocc.wait_all async [%1]
  rocc.wait_all [%1, %2]
  return
}
// CHECK-LABEL:   func.func @launch(
// CHECK:       affine.for %{{.*}} = 0 to 4 {
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:       affine.for %{{.*}} = 0 to 2 {
func.func @launch(%arg0: memref<16xf16>, %arg1: memref<16xf16>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  rocc.launch (%arg2, %arg3, %arg4) in (%arg5=%c4, %arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1) : memref<16xf16>, memref<16xf16> {
  }
  return
}