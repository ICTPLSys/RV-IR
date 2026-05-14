// Test case 1: Element-wise addition (tensor_tensor_add)
// Corresponds to: tensor_tensor_add(&tensor_A, &tensor_B, &tensor_C)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_elementwise_add(%arg0: memref<1x128x2048xf32>, %arg1: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<1x128x2048xf32>, memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.addf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
