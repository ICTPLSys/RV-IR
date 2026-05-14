// Test case 5: Element-wise exp (math.exp)
// Corresponds to: lut_exp(&tensor_in, &tensor_out)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_exp(%arg0: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
