// Test case: 4D Tensor exp operation with batch > 1
// Input: memref<32x32x128x128xf32> (4D: [batch=32, C=32, H=128, W=128])
// Output: memref<32x32x128x128xf32>
// Operation: element-wise exp
// Should generate: loop over batch dimension (32 iterations)

#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @test_exp_4d_batch32(%arg0: memref<32x32x128x128xf32>) -> memref<32x32x128x128xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32x128x128xf32>

    // Element-wise exp on 4D tensor
    linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<32x32x128x128xf32>)
      outs(%alloc : memref<32x32x128x128xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in : f32  
        linalg.yield %6 : f32
    }

    return %alloc : memref<32x32x128x128xf32>
  }
}
