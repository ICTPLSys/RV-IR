//./convert_riscv_mlir_to_c.sh tests/riscv_batch_matmul_test.mlir --strategy blocked
// #map = affine_map<(d0, d1, d2) -> (d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// module attributes {torch.debug_module_name = "LlamaDecoderBlock"} {
//   func.func @forward(%arg0: memref<1x256x256xf32>) -> memref<1x256x256xf32> {
//     %cst = arith.constant 0.0 : f32
//     %0 = arith.constant dense<2.0> : memref<256x256xf32>
//     %alloc_0 = memref.alloc() : memref<1x256x256xf32>
//     linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : memref<256x256xf32>) outs(%alloc_0 : memref<1x256x256xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     }
//     %alloc_1 = memref.alloc() : memref<1x256x256xf32>
//     linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x256x256xf32>)
//     rair.batch_matmul ins(%arg0, %alloc_0 : memref<1x256x256xf32>, memref<1x256x256xf32>) outs(%alloc_1 : memref<1x256x256xf32>)
//     return %alloc_1 : memref<1x256x256xf32>
//   }
// }
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "LlamaDecoderBlock"} {
  func.func @forward(%arg0: memref<1x128x2048xf32>) -> memref<1x128x512xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = arith.constant dense<2.0> : memref<512x2048xf32>

    %alloc_0 = memref.alloc() : memref<1x2048x512xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : memref<512x2048xf32>) outs(%alloc_0 : memref<1x2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }

    %alloc_1 = memref.alloc() : memref<1x128x512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x128x512xf32>)

    rair.batch_matmul ins(%arg0, %alloc_0 : memref<1x128x2048xf32>, memref<1x2048x512xf32>) outs(%alloc_1 : memref<1x128x512xf32>)

    return %alloc_1 : memref<1x128x512xf32>
  }
}
