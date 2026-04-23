// RUN: torch-mlir-opt <%s --convert-rocc-to-affine | FileCheck %s

// CHECK: affine.for %{{.*}} = 0 to 16 {
// CHECK:   affine.for %{{.*}} = 0 to 64 {
// CHECK:     affine.for %{{.*}} = 0 to 32 {
// CHECK:       arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:       arith.addf %{{.*}}, %{{.*}} : f32

module attributes {torch.debug_module_name = "Linear"} {
  memref.global "private" constant @__constant_32x64xf32 : memref<32x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}

  func.func @forward(%arg0: memref<16x32xf32, strided<[?, ?], offset: ?>>) -> memref<16x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_32x64xf32 : memref<32x64xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<16x64xf32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x64xf32>
    memref.copy %alloc_0, %alloc_1 : memref<16x64xf32> to memref<16x64xf32>
    rocc.matmul ins(%arg0, %0 : memref<16x32xf32, strided<[?, ?], offset: ?>>, memref<32x64xf32>) outs(%alloc_1 : memref<16x64xf32>)
    return %alloc_1 : memref<16x64xf32>
  }
}
