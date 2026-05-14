// Test case: Element-wise Division (arith.divf)
// Corresponds to: div_operator(&tensor_in1, &tensor_in2, &tensor_out)

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_elementwise_div(%arg0: memref<1x128x2048xf32>, %arg1: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    // Element-wise division
    %cst = arith.constant 2.048000e+03 : f32
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x2048xf32>, memref<1x128x2048xf32>) outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out: f32):
        %6 = arith.divf %in0, %in1 : f32
        linalg.yield %6 : f32
    }
    return %alloc : memref<1x128x2048xf32>
  }
}
