#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {torch.debug_module_name = "MnistNet"} {
  func.func @forward(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<32xf32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<32x196xf32>
    %cst_3 = arith.constant dense<[0.109545261, 0.150168076, 0.00724894181, -0.0129423486, 0.103974313, -0.153303191, 0.0109279193, 0.115677841, 0.12233936, 0.0208303425]> : tensor<10xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<10x32xf32>
    %0 = tensor.empty() : tensor<1x1x14x14xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x14x14xf32>) -> tensor<1x1x14x14xf32>
    %2 = tensor.empty() : tensor<2x2xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %2 : tensor<1x1x28x28xf32>, tensor<2x2xf32>) outs(%1 : tensor<1x1x14x14xf32>) -> tensor<1x1x14x14xf32>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2, 3]] : tensor<1x1x14x14xf32> into tensor<1x196xf32>
    %4 = tensor.empty() : tensor<196x32xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<32x196xf32>) outs(%4 : tensor<196x32xf32>) permutation = [1, 0] 
    %5 = tensor.empty() : tensor<1x32xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %7 = linalg.matmul ins(%collapsed, %transposed : tensor<1x196xf32>, tensor<196x32xf32>) outs(%6 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %cst : tensor<1x32xf32>, tensor<32xf32>) outs(%5 : tensor<1x32xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.addf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x32xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x32xf32>) outs(%5 : tensor<1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.cmpf ugt, %in, %cst_1 : f32
      %27 = arith.select %26, %in, %cst_1 : f32
      linalg.yield %27 : f32
    } -> tensor<1x32xf32>
    %10 = tensor.empty() : tensor<32x10xf32>
    %transposed_5 = linalg.transpose ins(%cst_4 : tensor<10x32xf32>) outs(%10 : tensor<32x10xf32>) permutation = [1, 0] 
    %11 = tensor.empty() : tensor<1x10xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %13 = linalg.matmul ins(%9, %transposed_5 : tensor<1x32xf32>, tensor<32x10xf32>) outs(%12 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %cst_3 : tensor<1x10xf32>, tensor<10xf32>) outs(%11 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.addf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x10xf32>
    %15 = tensor.empty() : tensor<1xi64>
    %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1xi64>) -> tensor<1xi64>
    %17 = tensor.empty() : tensor<1xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<1xf32>) -> tensor<1xf32>
    %19:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<1x10xf32>) outs(%18, %16 : tensor<1xf32>, tensor<1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_6: i64):
      %26 = linalg.index 1 : index
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.maximumf %in, %out : f32
      %29 = arith.cmpf ogt, %in, %out : f32
      %30 = arith.select %29, %27, %out_6 : i64
      linalg.yield %28, %30 : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)
    %expanded = tensor.expand_shape %19#0 [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %expanded : tensor<1x10xf32>, tensor<1x1xf32>) outs(%11 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.subf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x10xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1x10xf32>) outs(%11 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = math.exp %in : f32
      linalg.yield %26 : f32
    } -> tensor<1x10xf32>
    %22 = tensor.empty() : tensor<1x1xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%21 : tensor<1x10xf32>) outs(%23 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.addf %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<1x1xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%21, %24 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%11 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.divf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x10xf32>
    return %25 : tensor<1x10xf32>
  }
}
