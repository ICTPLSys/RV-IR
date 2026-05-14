// Test case: Identity/Copy operation with broadcast
// Input: memref<2048x8192xf32> (2D)
// Output: memref<1x2048x8192xf32> (3D - broadcast in first dimension)
// Corresponds to: broadcast_operator or memcpy

#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @test_identity_broadcast(%arg0: memref<2048x8192xf32>) -> memref<1x2048x8192xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2048x8192xf32>

    // Identity operation with broadcast
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<2048x8192xf32>)
      outs(%alloc : memref<1x2048x8192xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }

    return %alloc : memref<1x2048x8192xf32>
  }
}
