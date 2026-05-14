#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>

module {
  func.func @forward(%arg0: memref<1x32x32xf32>) -> memref<1x32x1xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc_4 = memref.alloc() : memref<1x32x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_4 : memref<1x32x1xf32>)
    linalg.generic {
      indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0 : memref<1x32x32xf32>) outs(%alloc_4 : memref<1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    }

    return %alloc_4 : memref<1x32x1xf32>
  }
}
