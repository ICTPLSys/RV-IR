// Simplified ResNet-style forward: one stem conv + BN + ReLU + global sum-pool + FC.
// Repeated weights use a single filled memref (no dense_resource / elided blobs).
// Note: this fork's --test-lower-to-llvm rejects `arith.constant dense<...> : memref<...>`;
// splat tensors use memref.alloc + linalg.fill with an explicit scalar (same values as dense<1.0>).
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "ResNetSimple"} {
  func.func @forward(%arg0: memref<1x3x4x4xf32>) -> memref<1x2xf32> {
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 1.000000e-05 : f64
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1.000000e+00 : f32

    %w_conv = memref.alloc() {alignment = 64 : i64} : memref<4x3x3x3xf32>
    linalg.fill ins(%c1 : f32) outs(%w_conv : memref<4x3x3x3xf32>)
    %bn_gamma = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%c1 : f32) outs(%bn_gamma : memref<4xf32>)
    %bn_beta = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%bn_beta : memref<4xf32>)
    %bn_mean = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%cst_2 : f32) outs(%bn_mean : memref<4xf32>)
    %bn_var = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    linalg.fill ins(%c1 : f32) outs(%bn_var : memref<4xf32>)

    %fc_w = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
    linalg.fill ins(%c1 : f32) outs(%fc_w : memref<2x4xf32>)
    %fc_b = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    linalg.fill ins(%cst_2 : f32) outs(%fc_b : memref<2xf32>)

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x10x10xf32>
    linalg.map outs(%alloc : memref<1x3x10x10xf32>)
      () {
        linalg.yield %cst_2 : f32
      }
    %subview = memref.subview %alloc[0, 0, 3, 3] [1, 3, 4, 4] [1, 1, 1, 1] : memref<1x3x10x10xf32> to memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x3x4x4xf32>) outs(%subview : memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>) {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a : f32
    }

    %conv_wrk = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.fill ins(%cst_2 : f32) outs(%conv_wrk : memref<1x4x8x8xf32>)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%alloc, %w_conv : memref<1x3x10x10xf32>, memref<4x3x3x3xf32>) outs(%conv_wrk : memref<1x4x8x8xf32>)

    %bn_out = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} 
    ins(%conv_wrk, %bn_gamma, %bn_beta, %bn_mean, %bn_var : memref<1x4x8x8xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>) outs(%bn_out : memref<1x4x8x8xf32>) {
    ^bb0(%in: f32, %in_g: f32, %in_b: f32, %in_m: f32, %in_v: f32, %out: f32):
      %eps = arith.truncf %cst_0 : f64 to f32
      %v = arith.addf %in_v, %eps : f32
      %rs = math.rsqrt %v : f32
      %d = arith.subf %in, %in_m : f32
      %n = arith.mulf %d, %rs : f32
      %s = arith.mulf %n, %in_g : f32
      %y = arith.addf %s, %in_b : f32
      linalg.yield %y : f32
    }

    %relu_out = memref.alloc() {alignment = 64 : i64} : memref<1x4x8x8xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bn_out : memref<1x4x8x8xf32>) outs(%relu_out : memref<1x4x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %p = arith.cmpf ugt, %in, %cst_2 : f32
      %q = arith.select %p, %in, %cst_2 : f32
      linalg.yield %q : f32
    }

    %pool_k = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
    linalg.fill ins(%c1 : f32) outs(%pool_k : memref<8x8xf32>)
    %pooled_wrk = memref.alloc() {alignment = 64 : i64} : memref<1x4x1x1xf32>
    linalg.fill ins(%cst_2 : f32) outs(%pooled_wrk : memref<1x4x1x1xf32>)
    linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<8> : vector<2xi64>} ins(%relu_out, %pool_k : memref<1x4x8x8xf32>, memref<8x8xf32>) outs(%pooled_wrk : memref<1x4x1x1xf32>)

    %pooled_avg = memref.alloc() {alignment = 64 : i64} : memref<1x4x1x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pooled_wrk : memref<1x4x1x1xf32>) outs(%pooled_avg : memref<1x4x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %d = arith.divf %in, %cst : f32
      linalg.yield %d : f32
    }

    %collapsed = memref.collapse_shape %pooled_avg [[0], [1, 2, 3]] : memref<1x4x1x1xf32> into memref<1x4xf32>
    %fc_w_t = memref.alloc() {alignment = 64 : i64} : memref<4x2xf32>
    linalg.transpose ins(%fc_w : memref<2x4xf32>) outs(%fc_w_t : memref<4x2xf32>) permutation = [1, 0]

    %logits = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    linalg.fill ins(%cst_2 : f32) outs(%logits : memref<1x2xf32>)
    linalg.matmul ins(%collapsed, %fc_w_t : memref<1x4xf32>, memref<4x2xf32>) outs(%logits : memref<1x2xf32>)

    %out = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%logits, %fc_b : memref<1x2xf32>, memref<2xf32>) outs(%out : memref<1x2xf32>) {
    ^bb0(%in: f32, %b: f32, %o: f32):
      %z = arith.addf %in, %b : f32
      linalg.yield %z : f32
    }
    return %out : memref<1x2xf32>
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %c1i = arith.constant 1 : i32
    %c52 = arith.constant 52 : i32
    %c1f = arith.constant 1.000000e+00 : f32
    %in = memref.alloc() {alignment = 64 : i64} : memref<1x3x4x4xf32>
    linalg.fill ins(%c1f : f32) outs(%in : memref<1x3x4x4xf32>)
    %res = func.call @forward(%in) : (memref<1x3x4x4xf32>) -> memref<1x2xf32>
    %v0 = memref.load %res[%c0, %c0] : memref<1x2xf32>
    %v1 = memref.load %res[%c0, %c1] : memref<1x2xf32>
    %i0 = arith.fptosi %v0 : f32 to i32
    %i1 = arith.fptosi %v1 : f32 to i32
    %sum = arith.addi %i0, %i1 : i32
    %ok = arith.cmpi eq, %sum, %c52 : i32
    %ret = arith.select %ok, %c0i, %c1i : i32
    return %ret : i32
  }
}
