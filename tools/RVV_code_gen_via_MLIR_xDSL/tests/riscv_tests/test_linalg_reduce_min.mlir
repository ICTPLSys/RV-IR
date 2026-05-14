// Test case: Reduce Min operation (arith.minimumf with reduction iterator)
// Corresponds to: reduce_dim0_min(&tensor_in, &tensor_out) under _npu_view_dims

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @test_reduce_min(%arg0: memref<1x128x2048xf32>) -> memref<1x128x1xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>

    // Reduce along last memref dimension (2048); NPU Tensor dim0
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0 : memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.minimumf %in, %out : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x1xf32>
  }
}
