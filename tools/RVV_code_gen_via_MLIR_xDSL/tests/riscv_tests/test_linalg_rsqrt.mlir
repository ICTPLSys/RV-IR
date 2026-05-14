// Test case 6: Element-wise rsqrt (math.rsqrt)
// Corresponds to: lut_squareroot(&tensor_in, &tensor_out) - note: NPU uses squareroot, need 1/sqrt
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_rsqrt(%arg0: memref<1x128x1xf32>) -> memref<1x128x1xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>

    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<1x128x1xf32>)
      outs(%alloc : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.rsqrt %in : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x1xf32>
  }
}
