// resnet_all_op: one forward that still exercises every op *kind* the full ResNet graph used
// (memref.{alloc,subview,copy}, linalg.{map,fill,generic,conv_2d_nchw_fchw,pooling_nchw_max,
// pooling_nchw_sum,transpose,matmul}, memref.collapse_shape, arith.* in region bodies),
// with tiny tensors so SPAD placement in linalg_mlir_to_c stays under per-bank limits.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "ResNetAllOpTiny"} {
  func.func @forward(%arg0: memref<1x3x4x4xf32>) -> memref<1x2xf32> {
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32

    // ---- weights (alloc + fill, same style as resnet_simple) ----
    %0 = memref.alloc() {alignment = 64 : i64} : memref<4x3x3x3xf32>
    linalg.fill ins(%c1 : f32) outs(%0 : memref<4x3x3x3xf32>)
    %1 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%c1 : f32) outs(%1 : memref<4xf32>)
    // 1x1 stride-2 projection (same op kind as full ResNet shortcut conv)
    %w_proj = memref.alloc() {alignment = 64 : i64} : memref<8x4x1x1xf32>
    linalg.fill ins(%c1 : f32) outs(%w_proj : memref<8x4x1x1xf32>)
    %2 = memref.alloc() {alignment = 64 : i64} : memref<8x8x3x3xf32>
    linalg.fill ins(%c1 : f32) outs(%2 : memref<8x8x3x3xf32>)
    %bn8 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    linalg.fill ins(%c1 : f32) outs(%bn8 : memref<8xf32>)
    %15 = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    linalg.fill ins(%cst_2 : f32) outs(%15 : memref<2xf32>)
    %16 = memref.alloc() {alignment = 64 : i64} : memref<2x8xf32>
    linalg.fill ins(%c1 : f32) outs(%16 : memref<2x8xf32>)

    // ---- pad + copy input (strided subview + memref.copy) ----
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x10x10xf32>
    linalg.map outs(%alloc : memref<1x3x10x10xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview = memref.subview %alloc[0, 0, 3, 3] [1, 3, 4, 4] [1, 1, 1, 1] : memref<1x3x10x10xf32> to memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>
    memref.copy %arg0, %subview : memref<1x3x4x4xf32> to memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>

    // ---- stem conv stride 1 (same as resnet_simple) ----
    %conv_wrk = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.fill ins(%cst_2 : f32) outs(%conv_wrk : memref<1x4x8x8xf32>)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc, %0 : memref<1x3x10x10xf32>, memref<4x3x3x3xf32>) outs(%conv_wrk : memref<1x4x8x8xf32>)

    %bn_out = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv_wrk, %1, %1, %1, %1 : memref<1x4x8x8xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%bn_out : memref<1x4x8x8xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }

    %relu_out = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bn_out : memref<1x4x8x8xf32>) outs(%relu_out : memref<1x4x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }

    // ---- max pool (negative padding fill + strided subview + copy + pooling_nchw_max) ----
    %pad_max = memref.alloc() {alignment = 64 : i64} : memref<1x4x10x10xf32>
    linalg.map outs(%pad_max : memref<1x4x10x10xf32>)
      () {
        linalg.yield %cst_1 : f32
      }
    %subview_max = memref.subview %pad_max[0, 0, 1, 1] [1, 4, 8, 8] [1, 1, 1, 1] : memref<1x4x10x10xf32> to memref<1x4x8x8xf32, strided<[400, 100, 10, 1], offset: 11>>
    memref.copy %relu_out, %subview_max : memref<1x4x8x8xf32> to memref<1x4x8x8xf32, strided<[400, 100, 10, 1], offset: 11>>
    %k_max = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    linalg.fill ins(%c1 : f32) outs(%k_max : memref<3x3xf32>)
    %after_max = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%after_max : memref<1x4x4x4xf32>)
    linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%pad_max, %k_max : memref<1x4x10x10xf32>, memref<3x3xf32>) outs(%after_max : memref<1x4x4x4xf32>)

    // ---- stride-2 1x1 projection conv (8x4x1x1) + BN + ReLU ----
    %proj_out = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%proj_out : memref<1x8x4x4xf32>)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%relu_out, %w_proj : memref<1x4x8x8xf32>, memref<8x4x1x1xf32>) outs(%proj_out : memref<1x8x4x4xf32>)
    %proj_bn = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%proj_out, %bn8, %bn8, %bn8, %bn8 : memref<1x8x4x4xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>) outs(%proj_bn : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %proj_relu = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%proj_bn : memref<1x8x4x4xf32>) outs(%proj_relu : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }

    // ---- padded 3x3 conv + residual add (3-input generic) ----
    %p2 = memref.alloc() {alignment = 64 : i64} : memref<1x8x6x6xf32>
    linalg.map outs(%p2 : memref<1x8x6x6xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %sv2 = memref.subview %p2[0, 0, 1, 1] [1, 8, 4, 4] [1, 1, 1, 1] : memref<1x8x6x6xf32> to memref<1x8x4x4xf32, strided<[288, 36, 6, 1], offset: 7>>
    memref.copy %proj_relu, %sv2 : memref<1x8x4x4xf32> to memref<1x8x4x4xf32, strided<[288, 36, 6, 1], offset: 7>>
    %conv2 = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%conv2 : memref<1x8x4x4xf32>)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%p2, %2 : memref<1x8x6x6xf32>, memref<8x8x3x3xf32>) outs(%conv2 : memref<1x8x4x4xf32>)
    %conv2_bn = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv2, %bn8, %bn8, %bn8, %bn8 : memref<1x8x4x4xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>) outs(%conv2_bn : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %in_118: f32, %in_119: f32, %in_120: f32, %in_121: f32, %out: f32):
      %17 = arith.truncf %cst_0 : f64 to f32
      %18 = arith.addf %in_121, %17 : f32
      %19 = math.rsqrt %18 : f32
      %20 = arith.subf %in, %in_120 : f32
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.mulf %21, %in_118 : f32
      %23 = arith.addf %22, %in_119 : f32
      linalg.yield %23 : f32
    }
    %conv2_relu = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv2_bn : memref<1x8x4x4xf32>) outs(%conv2_relu : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }
    %res_out = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv2_relu, %proj_relu : memref<1x8x4x4xf32>, memref<1x8x4x4xf32>) outs(%res_out : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    %res_relu = memref.alloc() {alignment = 64 : i64} : memref<1x8x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%res_out : memref<1x8x4x4xf32>) outs(%res_relu : memref<1x8x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.cmpf ugt, %in, %cst_2 : f32
      %18 = arith.select %17, %in, %cst_2 : f32
      linalg.yield %18 : f32
    }

    // ---- global sum pool + avg div + FC tail (collapse / transpose / matmul / bias) ----
    %pool_k = memref.alloc() {alignment = 64 : i64} : memref<4x4xf32>
    linalg.fill ins(%c1 : f32) outs(%pool_k : memref<4x4xf32>)
    %pooled_wrk = memref.alloc() {alignment = 64 : i64} : memref<1x8x1x1xf32>
    linalg.fill ins(%cst_2 : f32) outs(%pooled_wrk : memref<1x8x1x1xf32>)
    linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<4> : vector<2xi64>} ins(%res_relu, %pool_k : memref<1x8x4x4xf32>, memref<4x4xf32>) outs(%pooled_wrk : memref<1x8x1x1xf32>)
    %pooled_avg = memref.alloc() {alignment = 64 : i64} : memref<1x8x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pooled_wrk : memref<1x8x1x1xf32>) outs(%pooled_avg : memref<1x8x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %d = arith.divf %in, %cst : f32
      linalg.yield %d : f32
    }

    %collapsed = memref.collapse_shape %pooled_avg [[0], [1, 2, 3]] : memref<1x8x1x1xf32> into memref<1x8xf32>
    %fc_w_t = memref.alloc() {alignment = 64 : i64} : memref<8x2xf32>
    linalg.transpose ins(%16 : memref<2x8xf32>) outs(%fc_w_t : memref<8x2xf32>) permutation = [1, 0]

    %logits_wrk = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    linalg.fill ins(%cst_2 : f32) outs(%logits_wrk : memref<1x2xf32>)
    %logits = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    memref.copy %logits_wrk, %logits : memref<1x2xf32> to memref<1x2xf32>
    linalg.matmul ins(%collapsed, %fc_w_t : memref<1x8xf32>, memref<8x2xf32>) outs(%logits : memref<1x2xf32>)

    %out = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%logits, %15 : memref<1x2xf32>, memref<2xf32>) outs(%out : memref<1x2xf32>) {
    ^bb0(%in: f32, %in_118: f32, %out_0: f32):
      %17 = arith.addf %in, %in_118 : f32
      linalg.yield %17 : f32
    }
    return %out : memref<1x2xf32>
  }
}
