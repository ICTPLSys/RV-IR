// RUN: torch-mlir-opt <%s --convert-rocc-to-affine | FileCheck %s

// CHECK: affine.for %{{.*}} = 0 to 512 {
// CHECK:   affine.for %{{.*}} = 0 to 2048 {
// CHECK:     affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<512x2048xf32>
// CHECK:     affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2048x512xf32>
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "Linear"} {
  memref.global "private" constant @__constant_512x2048xf32 : memref<512x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  func.func @main(%arg0: memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x128x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_512x2048xf32 : memref<512x2048xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048x512xf32>
    rocc.transpose ins(%0 : memref<512x2048xf32>) outs(%alloc : memref<2048x512xf32>) {permutation = array<i64: 1, 0>}
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x512xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2048x512xf32>) outs(%alloc_0 : memref<1x2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x128x512xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
    memref.copy %alloc_1, %alloc_2 : memref<1x128x512xf32> to memref<1x128x512xf32>
    // rocc.batch_matmul ins(%arg0, %alloc_0 : memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>, memref<1x2048x512xf32>) outs(%alloc_2 : memref<1x128x512xf32>)
    return %alloc_2 : memref<1x128x512xf32>
    
  }
}






