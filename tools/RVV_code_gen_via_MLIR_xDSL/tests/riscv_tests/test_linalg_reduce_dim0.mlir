// Test case: Reduce along memref dim0 (outermost; maps to Tensor dim2)
// Corresponds to: reduce_dim2_sum/max/min under _npu_view_dims

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>

module {
  func.func @test_reduce_dim0_sum(%arg0: memref<128x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    // Reduce along first dimension (dim0)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel"]}
      ins(%arg0 : memref<128x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
