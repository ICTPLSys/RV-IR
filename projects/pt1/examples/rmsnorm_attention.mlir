#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> ()>
module attributes {torch.debug_module_name = "RMSNormAttention"} {
  func.func @forward(%arg0: tensor<1x2x16x32xf32>, %arg1: tensor<1x2x16x32xf32>, %arg2: tensor<1x2x16x32xf32>) -> tensor<1x2x16x32xf32> {
    %cst = arith.constant dense<9.9999999999999995E-7> : tensor<f64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 3.200000e+01 : f32
    %cst_2 = arith.constant dense<0.17677669529663689> : tensor<f64>
    %0 = tensor.empty() : tensor<1x2x16x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<1x2x16x32xf32>, tensor<1x2x16x32xf32>) outs(%0 : tensor<1x2x16x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %24 = arith.mulf %in, %in_7 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x32xf32>
    %2 = tensor.empty() : tensor<1x2x16x1xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x2x16x1xf32>) -> tensor<1x2x16x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1 : tensor<1x2x16x32xf32>) outs(%3 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = arith.addf %in, %out : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x2x16x1xf32>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = arith.divf %in, %cst_1 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %cst : tensor<1x2x16x1xf32>, tensor<f64>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %in_7: f64, %out: f32):
      %24 = arith.truncf %in_7 : f64 to f32
      %25 = arith.addf %in, %24 : f32
      linalg.yield %25 : f32
    } -> tensor<1x2x16x1xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x16x1xf32>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = math.sqrt %in : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %7 : tensor<1x2x16x32xf32>, tensor<1x2x16x1xf32>) outs(%0 : tensor<1x2x16x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %24 = arith.divf %in, %in_7 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x32xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg1 : tensor<1x2x16x32xf32>, tensor<1x2x16x32xf32>) outs(%0 : tensor<1x2x16x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %24 = arith.mulf %in, %in_7 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x32xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9 : tensor<1x2x16x32xf32>) outs(%3 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = arith.addf %in, %out : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x2x16x1xf32>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = arith.divf %in, %cst_1 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %cst : tensor<1x2x16x1xf32>, tensor<f64>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %in_7: f64, %out: f32):
      %24 = arith.truncf %in_7 : f64 to f32
      %25 = arith.addf %in, %24 : f32
      linalg.yield %25 : f32
    } -> tensor<1x2x16x1xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x2x16x1xf32>) outs(%2 : tensor<1x2x16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = math.sqrt %in : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x1xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %13 : tensor<1x2x16x32xf32>, tensor<1x2x16x1xf32>) outs(%0 : tensor<1x2x16x32xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %24 = arith.divf %in, %in_7 : f32
      linalg.yield %24 : f32
    } -> tensor<1x2x16x32xf32>
    %15 = tensor.empty() : tensor<1x2x32x16xf32>
    %transposed = linalg.transpose ins(%14 : tensor<1x2x16x32xf32>) outs(%15 : tensor<1x2x32x16xf32>) permutation = [0, 1, 3, 2] 
    %collapsed = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x2x16x32xf32> into tensor<2x16x32xf32>
    %collapsed_3 = tensor.collapse_shape %transposed [[0, 1], [2], [3]] : tensor<1x2x32x16xf32> into tensor<2x32x16xf32>
    %16 = tensor.empty() : tensor<2x16x16xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %18 = linalg.batch_matmul ins(%collapsed, %collapsed_3 : tensor<2x16x32xf32>, tensor<2x32x16xf32>) outs(%17 : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %expanded = tensor.expand_shape %18 [[0, 1], [2], [3]] output_shape [1, 2, 16, 16] : tensor<2x16x16xf32> into tensor<1x2x16x16xf32>
    %19 = tensor.empty() : tensor<1x2x16x16xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %cst_2 : tensor<1x2x16x16xf32>, tensor<f64>) outs(%19 : tensor<1x2x16x16xf32>) {
    ^bb0(%in: f32, %in_7: f64, %out: f32):
      %24 = arith.truncf %in_7 : f64 to f32
      %25 = arith.mulf %in, %24 : f32
      linalg.yield %25 : f32
    } -> tensor<1x2x16x16xf32>
    %collapsed_4 = tensor.collapse_shape %20 [[0, 1], [2], [3]] : tensor<1x2x16x16xf32> into tensor<2x16x16xf32>
    %collapsed_5 = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<1x2x16x32xf32> into tensor<2x16x32xf32>
    %21 = tensor.empty() : tensor<2x16x32xf32>
    %22 = linalg.fill ins(%cst_0 : f32) outs(%21 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
    %23 = linalg.batch_matmul ins(%collapsed_4, %collapsed_5 : tensor<2x16x16xf32>, tensor<2x16x32xf32>) outs(%22 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
    %expanded_6 = tensor.expand_shape %23 [[0, 1], [2], [3]] output_shape [1, 2, 16, 32] : tensor<2x16x32xf32> into tensor<1x2x16x32xf32>
    return %expanded_6 : tensor<1x2x16x32xf32>
  }
}
