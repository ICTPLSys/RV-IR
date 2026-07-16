#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module attributes {torch.debug_module_name = "LlamaDecoderBlock"} {
  // memref.global "private" constant @__constant_2048x8192xf32 : memref<2048x8192xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_8192x2048xf32 : memref<8192x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_512x2048xf32 : memref<512x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_2048x2048xf32 : memref<2048x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_xf64 : memref<f64> = dense<1.000000e-05> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_2048xf32 : memref<2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  func.func @forward(%arg0: memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x128x2048xf32> {
    %cst = arith.constant 2.048000e+03 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %0 = arith.constant dense<1.0> : memref<2048xf32>
    %1 = arith.constant dense<1.0> : memref<f64>
    %2 = arith.constant dense<1.0> : memref<2048x2048xf32>
    %3 = arith.constant dense<1.0> : memref<512x2048xf32>
    %4 = arith.constant dense<1.0> : memref<8192x2048xf32>
    %5 = arith.constant dense<1.0> : memref<2048x8192xf32>
    %alloc = arith.constant dense<1.0> : memref<1x128x2048xf32>
    // %0 = memref.get_global @__constant_2048xf32 : memref<2048xf32>
    // %1 = memref.get_global @__constant_xf64 : memref<f64>
    // %2 = memref.get_global @__constant_2048x2048xf32 : memref<2048x2048xf32>
    // %3 = memref.get_global @__constant_512x2048xf32 : memref<512x2048xf32>
    // %4 = memref.get_global @__constant_8192x2048xf32 : memref<8192x2048xf32>
    // %5 = memref.get_global @__constant_2048x8192xf32 : memref<2048x8192xf32>
    // %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %6 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_3 : memref<1x128x1xf32>)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_3, %alloc_4 : memref<1x128x1xf32> to memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc : memref<1x128x2048xf32>) outs(%alloc_4 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_4 : memref<1x128x1xf32>) outs(%alloc_5 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.divf %in, %cst : f32
      linalg.yield %6 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_5, %1 : memref<1x128x1xf32>, memref<f64>) outs(%alloc_6 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %in_76: f64, %out: f32):
      %6 = arith.truncf %in_76 : f64 to f32
      %7 = arith.addf %in, %6 : f32
      linalg.yield %7 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_6 : memref<1x128x1xf32>) outs(%alloc_7 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.rsqrt %in : f32
      linalg.yield %6 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc_7 : memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>, memref<1x128x1xf32>) outs(%alloc_8 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map3, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %alloc_8 : memref<2048xf32>, memref<1x128x2048xf32>) outs(%alloc_9 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32>
    linalg.transpose ins(%2 : memref<2048x2048xf32>) outs(%alloc_10 : memref<2048x2048xf32>) permutation = [1, 0] 
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x2048xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_10 : memref<2048x2048xf32>) outs(%alloc_11 : memref<1x2048x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_12 : memref<1x128x2048xf32>)
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    memref.copy %alloc_12, %alloc_13 : memref<1x128x2048xf32> to memref<1x128x2048xf32>
    rair.batch_matmul ins(%alloc_9, %alloc_11 : memref<1x128x2048xf32>, memref<1x2048x2048xf32>) outs(%alloc_13 : memref<1x128x2048xf32>)
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<2048x512xf32>
   linalg.transpose ins(%3 : memref<512x2048xf32>) outs(%alloc_14 : memref<2048x512xf32>) permutation = [1, 0] 
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x512xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_14 : memref<2048x512xf32>) outs(%alloc_15 : memref<1x2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }  
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_16 : memref<1x128x512xf32>)
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
    memref.copy %alloc_16, %alloc_17 : memref<1x128x512xf32> to memref<1x128x512xf32>
    rair.batch_matmul ins(%alloc_9, %alloc_15 : memref<1x128x2048xf32>, memref<1x2048x512xf32>) outs(%alloc_17 : memref<1x128x512xf32>)
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<2048x512xf32>
   linalg.transpose ins(%3 : memref<512x2048xf32>) outs(%alloc_18 : memref<2048x512xf32>) permutation = [1, 0] 
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x512xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_18 : memref<2048x512xf32>) outs(%alloc_19 : memref<1x2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
    memref.copy %alloc_16, %alloc_20 : memref<1x128x512xf32> to memref<1x128x512xf32>
    rair.batch_matmul ins(%alloc_9, %alloc_19 : memref<1x128x2048xf32>, memref<1x2048x512xf32>) outs(%alloc_20 : memref<1x128x512xf32>)
    %expand_shape = memref.expand_shape %alloc_13 [[0], [1], [2, 3]] output_shape [1, 128, 32, 64] : memref<1x128x2048xf32> into memref<1x128x32x64xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x64xf32>
    linalg.transpose ins(%expand_shape : memref<1x128x32x64xf32>) outs(%alloc_21 : memref<1x32x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expand_shape_22 = memref.expand_shape %alloc_17 [[0], [1], [2, 3]] output_shape [1, 128, 8, 64] : memref<1x128x512xf32> into memref<1x128x8x64xf32>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x8x128x64xf32>
    linalg.transpose ins(%expand_shape_22 : memref<1x128x8x64xf32>) outs(%alloc_23 : memref<1x8x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expand_shape_24 = memref.expand_shape %alloc_20 [[0], [1], [2, 3]] output_shape [1, 128, 8, 64] : memref<1x128x512xf32> into memref<1x128x8x64xf32>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1x8x128x64xf32>
    linalg.transpose ins(%expand_shape_24 : memref<1x128x8x64xf32>) outs(%alloc_25 : memref<1x8x128x64xf32>) permutation = [0, 2, 1, 3] 
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x128x64xf32>
    linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc_23 : memref<1x8x128x64xf32>) outs(%alloc_26 : memref<1x8x4x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %collapse_shape = memref.collapse_shape %alloc_26 [[0], [1, 2], [3], [4]] : memref<1x8x4x128x64xf32> into memref<1x32x128x64xf32>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x128x64xf32>
    linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc_25 : memref<1x8x128x64xf32>) outs(%alloc_27 : memref<1x8x4x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1x32x64x128xf32>
    linalg.transpose ins(%collapse_shape : memref<1x32x128x64xf32>) outs(%alloc_28 : memref<1x32x64x128xf32>) permutation = [0, 1, 3, 2] 
    %collapse_shape_29 = memref.collapse_shape %alloc_21 [[0, 1], [2], [3]] : memref<1x32x128x64xf32> into memref<32x128x64xf32>
    %collapse_shape_30 = memref.collapse_shape %alloc_28 [[0, 1], [2], [3]] : memref<1x32x64x128xf32> into memref<32x64x128xf32>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<32x128x128xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_31 : memref<32x128x128xf32>)
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<32x128x128xf32>
    memref.copy %alloc_31, %alloc_32 : memref<32x128x128xf32> to memref<32x128x128xf32>
    rair.batch_matmul ins(%collapse_shape_29, %collapse_shape_30 : memref<32x128x64xf32>, memref<32x64x128xf32>) outs(%alloc_32 : memref<32x128x128xf32>)
    %expand_shape_33 = memref.expand_shape %alloc_32 [[0, 1], [2], [3]] output_shape [1, 32, 128, 128] : memref<32x128x128xf32> into memref<1x32x128x128xf32>
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128xi64>
    linalg.fill ins(%c0_i64 : i64) outs(%alloc_34 : memref<1x32x128xi64>)
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_35 : memref<1x32x128xf32>)
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128xf32>
    memref.copy %alloc_35, %alloc_36 : memref<1x32x128xf32> to memref<1x32x128xf32>
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128xi64>
    memref.copy %alloc_34, %alloc_37 : memref<1x32x128xi64> to memref<1x32x128xi64>
    linalg.generic {indexing_maps = [#map7, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_shape_33 : memref<1x32x128x128xf32>) outs(%alloc_36, %alloc_37 : memref<1x32x128xf32>, memref<1x32x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_76: i64):
      %6 = linalg.index 3 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = arith.maximumf %in, %out : f32
      %9 = arith.cmpf ogt, %in, %out : f32
      %10 = arith.select %9, %7, %out_76 : i64
      linalg.yield %8, %10 : f32, i64
    }
    %expand_shape_38 = memref.expand_shape %alloc_36 [[0], [1], [2, 3]] output_shape [1, 32, 128, 1] : memref<1x32x128xf32> into memref<1x32x128x1xf32>
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x128xf32>
    linalg.generic {indexing_maps = [#map7, #map9, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape_33, %expand_shape_38 : memref<1x32x128x128xf32>, memref<1x32x128x1xf32>) outs(%alloc_39 : memref<1x32x128x128xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.subf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x128xf32>
    linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_39 : memref<1x32x128x128xf32>) outs(%alloc_40 : memref<1x32x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.exp %in : f32
      linalg.yield %6 : f32
    }
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x1xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_41 : memref<1x32x128x1xf32>)
    %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x1xf32>
    memref.copy %alloc_41, %alloc_42 : memref<1x32x128x1xf32> to memref<1x32x128x1xf32>
    linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_40 : memref<1x32x128x128xf32>) outs(%alloc_42 : memref<1x32x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    }
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<1x32x128x128xf32>
    linalg.generic {indexing_maps = [#map7, #map9, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_40, %alloc_42 : memref<1x32x128x128xf32>, memref<1x32x128x1xf32>) outs(%alloc_43 : memref<1x32x128x128xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.divf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %collapse_shape_44 = memref.collapse_shape %alloc_43 [[0, 1], [2], [3]] : memref<1x32x128x128xf32> into memref<32x128x128xf32>
    %collapse_shape_45 = memref.collapse_shape %alloc_27 [[0, 1, 2], [3], [4]] : memref<1x8x4x128x64xf32> into memref<32x128x64xf32>
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<32x128x64xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_46 : memref<32x128x64xf32>)
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<32x128x64xf32>
    memref.copy %alloc_46, %alloc_47 : memref<32x128x64xf32> to memref<32x128x64xf32>
    rair.batch_matmul ins(%collapse_shape_44, %collapse_shape_45 : memref<32x128x128xf32>, memref<32x128x64xf32>) outs(%alloc_47 : memref<32x128x64xf32>)
    %expand_shape_48 = memref.expand_shape %alloc_47 [[0, 1], [2], [3]] output_shape [1, 32, 128, 64] : memref<32x128x64xf32> into memref<1x32x128x64xf32>
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x128x32x64xf32>
    linalg.transpose ins(%expand_shape_48 : memref<1x32x128x64xf32>) outs(%alloc_49 : memref<1x128x32x64xf32>) permutation = [0, 2, 1, 3] 
    %collapse_shape_50 = memref.collapse_shape %alloc_49 [[0], [1], [2, 3]] : memref<1x128x32x64xf32> into memref<1x128x2048xf32>
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32>
    linalg.transpose ins(%2 : memref<2048x2048xf32>) outs(%alloc_51 : memref<2048x2048xf32>) permutation = [1, 0] 
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x2048xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_51 : memref<2048x2048xf32>) outs(%alloc_52 : memref<1x2048x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    memref.copy %alloc_12, %alloc_53 : memref<1x128x2048xf32> to memref<1x128x2048xf32>
    rair.batch_matmul ins(%collapse_shape_50, %alloc_52 : memref<1x128x2048xf32>, memref<1x2048x2048xf32>) outs(%alloc_53 : memref<1x128x2048xf32>)
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %alloc_53 : memref<1x128x2048xf32, strided<[?, ?, ?], offset: ?>>, memref<1x128x2048xf32>) outs(%alloc_54 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.addf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_54 : memref<1x128x2048xf32>) outs(%alloc_55 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %6 : f32
    }
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_3, %alloc_56 : memref<1x128x1xf32> to memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_55 : memref<1x128x2048xf32>) outs(%alloc_56 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    }
    %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_56 : memref<1x128x1xf32>) outs(%alloc_57 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.divf %in, %cst : f32
      linalg.yield %6 : f32
    }

    %alloc_58 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_57, %1 : memref<1x128x1xf32>, memref<f64>) outs(%alloc_58 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %in_76: f64, %out: f32):
      %6 = arith.truncf %in_76 : f64 to f32
      %7 = arith.addf %in, %6 : f32
      linalg.yield %7 : f32
    }
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_58 : memref<1x128x1xf32>) outs(%alloc_59 : memref<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.rsqrt %in : f32
      linalg.yield %6 : f32
    }
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_54, %alloc_59 : memref<1x128x2048xf32>, memref<1x128x1xf32>) outs(%alloc_60 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_61 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map3, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %alloc_60 : memref<2048xf32>, memref<1x128x2048xf32>) outs(%alloc_61 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<2048x8192xf32>
   linalg.transpose ins(%4 : memref<8192x2048xf32>) outs(%alloc_62 : memref<2048x8192xf32>) permutation = [1, 0] 
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x8192xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_62 : memref<2048x8192xf32>) outs(%alloc_63 : memref<1x2048x8192xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    linalg.fill ins(%cst_2 : f32) outs(%alloc_64 : memref<1x128x8192xf32>)
    %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    memref.copy %alloc_64, %alloc_65 : memref<1x128x8192xf32> to memref<1x128x8192xf32>
    rair.batch_matmul ins(%alloc_61, %alloc_63 : memref<1x128x2048xf32>, memref<1x2048x8192xf32>) outs(%alloc_65 : memref<1x128x8192xf32>)
    %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_65 : memref<1x128x8192xf32>) outs(%alloc_66 : memref<1x128x8192xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.negf %in : f32
      %7 = math.exp %6 : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = arith.divf %cst_0, %8 : f32
      linalg.yield %9 : f32
    }
    %alloc_67 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_66, %alloc_65 : memref<1x128x8192xf32>, memref<1x128x8192xf32>) outs(%alloc_67 : memref<1x128x8192xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_68 = memref.alloc() {alignment = 64 : i64} : memref<2048x8192xf32>
   linalg.transpose ins(%4 : memref<8192x2048xf32>) outs(%alloc_68 : memref<2048x8192xf32>) permutation = [1, 0] 
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<1x2048x8192xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_68 : memref<2048x8192xf32>) outs(%alloc_69 : memref<1x2048x8192xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    memref.copy %alloc_64, %alloc_70 : memref<1x128x8192xf32> to memref<1x128x8192xf32>
    rair.batch_matmul ins(%alloc_61, %alloc_69 : memref<1x128x2048xf32>, memref<1x2048x8192xf32>) outs(%alloc_70 : memref<1x128x8192xf32>)
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x128x8192xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_67, %alloc_70 : memref<1x128x8192xf32>, memref<1x128x8192xf32>) outs(%alloc_71 : memref<1x128x8192xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.mulf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<8192x2048xf32>
   linalg.transpose ins(%5 : memref<2048x8192xf32>) outs(%alloc_72 : memref<8192x2048xf32>) permutation = [1, 0] 
    %alloc_73 = memref.alloc() {alignment = 64 : i64} : memref<1x8192x2048xf32>
    linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_72 : memref<8192x2048xf32>) outs(%alloc_73 : memref<1x8192x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    memref.copy %alloc_12, %alloc_74 : memref<1x128x2048xf32> to memref<1x128x2048xf32>
    rair.batch_matmul ins(%alloc_71, %alloc_73 : memref<1x128x8192xf32>, memref<1x8192x2048xf32>) outs(%alloc_74 : memref<1x128x2048xf32>)
    %alloc_75 = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_54, %alloc_74 : memref<1x128x2048xf32>, memref<1x128x2048xf32>) outs(%alloc_75 : memref<1x128x2048xf32>) {
    ^bb0(%in: f32, %in_76: f32, %out: f32):
      %6 = arith.addf %in, %in_76 : f32
      linalg.yield %6 : f32
    }
    return %alloc_75 : memref<1x128x2048xf32>
  }
}
