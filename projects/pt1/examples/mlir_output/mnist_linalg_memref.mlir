#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {torch.debug_module_name = "MnistNet"} {
  func.func @forward(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32x196xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_2 : memref<32x196xf32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_3 : memref<10xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<10x32xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_4 : memref<10x32xf32>)
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14x14xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<1x1x14x14xf32>)
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x1x14x14xf32>
    memref.copy %alloc_5, %alloc_7 : memref<1x1x14x14xf32> to memref<1x1x14x14xf32>
    linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %alloc_6 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<2x2xf32>) outs(%alloc_7 : memref<1x1x14x14xf32>)
    %collapse_shape = memref.collapse_shape %alloc_7 [[0], [1, 2, 3]] : memref<1x1x14x14xf32> into memref<1x196xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<196x32xf32>
    linalg.transpose ins(%alloc_2 : memref<32x196xf32>) outs(%alloc_8 : memref<196x32xf32>) permutation = [1, 0] 
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_9 : memref<1x32xf32>)
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32>
    memref.copy %alloc_9, %alloc_10 : memref<1x32xf32> to memref<1x32xf32>
    linalg.matmul ins(%collapse_shape, %alloc_8 : memref<1x196xf32>, memref<196x32xf32>) outs(%alloc_10 : memref<1x32xf32>)
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_10, %alloc : memref<1x32xf32>, memref<32xf32>) outs(%alloc_11 : memref<1x32xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %0 = arith.addf %in, %in_26 : f32
      linalg.yield %0 : f32
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x32xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_11 : memref<1x32xf32>) outs(%alloc_12 : memref<1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ugt, %in, %cst : f32
      %1 = arith.select %0, %in, %cst : f32
      linalg.yield %1 : f32
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<32x10xf32>
    linalg.transpose ins(%alloc_4 : memref<10x32xf32>) outs(%alloc_13 : memref<32x10xf32>) permutation = [1, 0] 
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_14 : memref<1x10xf32>)
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    memref.copy %alloc_14, %alloc_15 : memref<1x10xf32> to memref<1x10xf32>
    linalg.matmul ins(%alloc_12, %alloc_13 : memref<1x32xf32>, memref<32x10xf32>) outs(%alloc_15 : memref<1x10xf32>)
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_3 : memref<1x10xf32>, memref<10xf32>) outs(%alloc_16 : memref<1x10xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %0 = arith.addf %in, %in_26 : f32
      linalg.yield %0 : f32
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_17 : memref<1xi64>)
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_18 : memref<1xf32>)
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    memref.copy %alloc_18, %alloc_19 : memref<1xf32> to memref<1xf32>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
    memref.copy %alloc_17, %alloc_20 : memref<1xi64> to memref<1xi64>
    linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%alloc_16 : memref<1x10xf32>) outs(%alloc_19, %alloc_20 : memref<1xf32>, memref<1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_26: i64):
      %0 = linalg.index 1 : index
      %1 = arith.index_cast %0 : index to i64
      %2 = arith.maximumf %in, %out : f32
      %3 = arith.cmpf ogt, %in, %out : f32
      %4 = arith.select %3, %1, %out_26 : i64
      linalg.yield %2, %4 : f32, i64
    }
    %expand_shape = memref.expand_shape %alloc_19 [[0, 1]] output_shape [1, 1] : memref<1xf32> into memref<1x1xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_16, %expand_shape : memref<1x10xf32>, memref<1x1xf32>) outs(%alloc_21 : memref<1x10xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %0 = arith.subf %in, %in_26 : f32
      linalg.yield %0 : f32
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_21 : memref<1x10xf32>) outs(%alloc_22 : memref<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.exp %in : f32
      linalg.yield %0 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_23 : memref<1x1xf32>)
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x1xf32>
    memref.copy %alloc_23, %alloc_24 : memref<1x1xf32> to memref<1x1xf32>
    linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%alloc_22 : memref<1x10xf32>) outs(%alloc_24 : memref<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_22, %alloc_24 : memref<1x10xf32>, memref<1x1xf32>) outs(%alloc_25 : memref<1x10xf32>) {
    ^bb0(%in: f32, %in_26: f32, %out: f32):
      %0 = arith.divf %in, %in_26 : f32
      linalg.yield %0 : f32
    }
    return %alloc_25 : memref<1x10xf32>
  }
}
