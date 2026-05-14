// RMSNorm Test Case
// This file tests the RMSNorm operation using the expanded form
// Formula: output = input / sqrt(mean(input^2) + epsilon) * gamma
//
// The RMSNorm operation is expanded into 7 steps:
// 1. Square: x^2
// 2. Reduce Sum: sum(x^2, axis=-1)
// 3. Divide by dim: sum / dim_size
// 4. Add epsilon: mean + eps
// 5. Rsqrt: 1 / sqrt(...)
// 6. Multiply by input: x * rsqrt
// 7. Multiply by gamma: result * gamma

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
  // Global constant for gamma (example)
module attributes {torch.debug_module_name = "RMSNormTest"} {
  func.func @rmsnorm_test(%arg0: memref<1x128x2048xf32>) -> memref<1x128x2048xf32> {
    // Constants
    %cst = arith.constant 2.048000e+03 : f32  // hidden_dim = 2048
    %cst_0 = arith.constant dense<1.000000e-05> : memref<f64> // epsilon = 1e-5
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32

    // Gamma (scale) parameter - loaded from weights
    // %gamma = memref.get_global @__constant_2048xf32 : memref<2048xf32>
    %gamma = arith.constant dense<1.0> : memref<2048xf32>
    // Step 1: Square (x^2)
    %alloc_square = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : memref<1x128x2048xf32>) outs(%alloc_square : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %out: f32): 
        %6 = math.fpowi %in, %c2_i64 : f32, i64
        linalg.yield %6 : f32
    }

    // Step 2: Reduce Sum along last dimension
    %alloc_sum = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc_sum : memref<1x128x1xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%alloc_square : memref<1x128x2048xf32>) outs(%alloc_sum : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
    }

    // Step 3: Divide by dimension (mean of squares)
    %alloc_mean = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%alloc_sum : memref<1x128x1xf32>) outs(%alloc_mean : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.divf %in, %cst : f32
        linalg.yield %6 : f32
    }

    // Step 4: Add epsilon
    %alloc_eps = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%alloc_mean, %cst_0 : memref<1x128x1xf32>, memref<f64>) outs(%alloc_eps : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %in_76: f64, %out: f32):
        %6 = arith.truncf %in_76 : f64 to f32
        %7 = arith.addf %in, %6 : f32
        linalg.yield %7 : f32
    }

    // Step 5: Rsqrt (1 / sqrt(...))
    %alloc_rsqrt = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%alloc_eps : memref<1x128x1xf32>) outs(%alloc_rsqrt : memref<1x128x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.rsqrt %in : f32
        linalg.yield %6 : f32
    }

    // Step 6: Multiply by input (x * rsqrt)
    %alloc_normalized = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %alloc_rsqrt : memref<1x128x2048xf32>, memref<1x128x1xf32>) outs(%alloc_normalized : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.mulf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    // Step 7: Multiply by gamma
    %alloc_output = memref.alloc() {alignment = 64 : i64} : memref<1x128x2048xf32>
    linalg.generic {indexing_maps = [#map3, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%gamma, %alloc_normalized : memref<2048xf32>, memref<1x128x2048xf32>) outs(%alloc_output : memref<1x128x2048xf32>) {
      ^bb0(%in: f32, %in_76: f32, %out: f32):
        %6 = arith.mulf %in, %in_76 : f32
        linalg.yield %6 : f32
    }

    return %alloc_output : memref<1x128x2048xf32>
  }


}
