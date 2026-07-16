#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "ResNetSimple"} {
  func.func @forward(%arg0: memref<1x3x5x5xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x4xf32> {
    %cst = arith.constant 8.100000e+01 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc : memref<4xf32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<4x3x3x3xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_3 : memref<4x3x3x3xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_4 : memref<4xf32>)
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_5 : memref<4xf32>)
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_6 : memref<4xf32>)
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<4x4xf32>)
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x3x11x11xf32>
    linalg.map outs(%alloc_8 : memref<1x3x11x11xf32>)
      () {
        linalg.yield %cst_1 : f32
      }
    %subview = memref.subview %alloc_8[0, 0, 3, 3] [1, 3, 5, 5] [1, 1, 1, 1] : memref<1x3x11x11xf32> to memref<1x3x5x5xf32, strided<[363, 121, 11, 1], offset: 36>>
    memref.copy %arg0, %subview : memref<1x3x5x5xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x3x5x5xf32, strided<[363, 121, 11, 1], offset: 36>>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x4x9x9xf32>
    linalg.broadcast ins(%alloc : memref<4xf32>) outs(%alloc_9 : memref<1x4x9x9xf32>) dimensions = [0, 2, 3] 
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x4x9x9xf32>
    memref.copy %alloc_9, %alloc_10 : memref<1x4x9x9xf32> to memref<1x4x9x9xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc_8, %alloc_3 : memref<1x3x11x11xf32>, memref<4x3x3x3xf32>) outs(%alloc_10 : memref<1x4x9x9xf32>)
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x4x9x9xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_10, %alloc_4, %alloc_5, %alloc_5, %alloc_4 : memref<1x4x9x9xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%alloc_11 : memref<1x4x9x9xf32>) {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %0 = arith.truncf %cst_0 : f64 to f32
      %1 = arith.addf %in_23, %0 : f32
      %2 = math.rsqrt %1 : f32
      %3 = arith.subf %in, %in_22 : f32
      %4 = arith.mulf %3, %2 : f32
      %5 = arith.mulf %4, %in_20 : f32
      %6 = arith.addf %5, %in_21 : f32
      linalg.yield %6 : f32
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x4x9x9xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_11 : memref<1x4x9x9xf32>) outs(%alloc_12 : memref<1x4x9x9xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ugt, %in, %cst_1 : f32
      %1 = arith.select %0, %in, %cst_1 : f32
      linalg.yield %1 : f32
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x1x1xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_13 : memref<1x4x1x1xf32>)
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<9x9xf32>
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x4x1x1xf32>
    memref.copy %alloc_13, %alloc_15 : memref<1x4x1x1xf32> to memref<1x4x1x1xf32>
    linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<9> : vector<2xi64>} ins(%alloc_12, %alloc_14 : memref<1x4x9x9xf32>, memref<9x9xf32>) outs(%alloc_15 : memref<1x4x1x1xf32>)
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x4x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_15 : memref<1x4x1x1xf32>) outs(%alloc_16 : memref<1x4x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.divf %in, %cst : f32
      linalg.yield %0 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_16 [[0], [1, 2, 3]] : memref<1x4x1x1xf32> into memref<1x4xf32>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_17 : memref<1x4xf32>)
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    memref.copy %alloc_17, %alloc_18 : memref<1x4xf32> to memref<1x4xf32>
    rair.matmul ins(%collapse_shape, %alloc_7 : memref<1x4xf32>, memref<4x4xf32>) outs(%alloc_18 : memref<1x4xf32>)
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc_18, %alloc_6 : memref<1x4xf32>, memref<4xf32>) outs(%alloc_19 : memref<1x4xf32>) {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %0 = arith.addf %in, %in_20 : f32
      linalg.yield %0 : f32
    }
    return %alloc_19 : memref<1x4xf32>
  }
}

