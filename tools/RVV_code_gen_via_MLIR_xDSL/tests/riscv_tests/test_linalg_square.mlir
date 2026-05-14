// Test case 3: Square operation (math.fpowi with exponent 2)
// Corresponds to: square_operator(&tensor_in, &tensor_out)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_square(%arg0: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    %c2_i64 = arith.constant 2 : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>

    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<1x128x2048xf32>)
      outs(%alloc : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.fpowi %in, %c2_i64 : f32, i64
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x2048xf32>
  }
}
