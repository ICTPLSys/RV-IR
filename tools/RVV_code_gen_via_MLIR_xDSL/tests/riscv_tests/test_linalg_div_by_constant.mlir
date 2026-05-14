// Test case: Division by constant (scalar)
// This should be converted to:
// 1. Create a constant tensor with the same shape as input
// 2. Fill it with the constant value
// 3. Call div_operator(&tensor_in, &tensor_constant, &tensor_out)

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @test_div_by_constant(%arg0: memref<1x128x1xf32>) -> memref<1x128x1xf32> {
    %cst = arith.constant 0.5 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>

    // Division by constant
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<1x128x1xf32>)
      outs(%alloc : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.divf %in, %cst : f32
        linalg.yield %6 : f32
    }

    return %alloc : memref<1x128x1xf32>
  }
}
