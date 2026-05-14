// Test case: 4D Tensor subtraction operation
// Input1: memref<1x32x128x128xf32> (4D: [batch=1, C=32, H=128, W=128])
// Input2: memref<1x32x128x1xf32> (4D: [batch=1, C=32, H=128, W=1])
// Output: memref<1x32x128x128xf32>
// Operation: element-wise subtraction (broadcast on last dimension)
// Should generate: loop over batch dimension, calling tensor_tensor_sub for each 3D slice

#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @test_sub_4d_broadcast(%arg0: memref<1x32x128x128xf32>, %arg1: memref<1x32x128x1xf32>) -> memref<1x32x128x128xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x128xf32>

    // Element-wise subtraction with broadcasting
    linalg.generic {indexing_maps = [#map7, #map9, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<1x32x128x128xf32>, memref<1x32x128x1xf32>)
      outs(%alloc : memref<1x32x128x128xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.subf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x32x128x128xf32>
  }
}
