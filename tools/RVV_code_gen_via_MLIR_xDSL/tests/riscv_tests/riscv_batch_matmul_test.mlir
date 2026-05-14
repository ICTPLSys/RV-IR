//./convert_riscv_mlir_to_c.sh tests/riscv_batch_matmul_test.mlir --strategy simple
//./convert_riscv_mlir_to_c.sh tests/riscv_batch_matmul_test.mlir --strategy workload
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "Linear"} {
  func.func @forward(%arg0: memref<1x4x8xf32>) -> memref<1x4x16xf32> {
    %cst = arith.constant 0.0 : f32
    %0 =  arith.constant  dense<2.0> : memref<16x8xf32>
    // %alloc = memref.alloc() : memref<8x16xf32>
    %alloc =  arith.constant  dense<2.0> : memref<8x16xf32>
    // rocc.transpose ins(%0 : memref<16x8xf32>) outs(%alloc : memref<8x16xf32>) {permutation = array<i64: 1, 0>}
    %alloc_0 = memref.alloc() : memref<1x8x16xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<8x16xf32>) outs(%alloc_0 : memref<1x8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_1 = memref.alloc() : memref<1x4x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x4x16xf32>)
    rocc.batch_matmul ins(%arg0, %alloc_0 : memref<1x4x8xf32>, memref<1x8x16xf32>) outs(%alloc_1 : memref<1x4x16xf32>)
    return %alloc_1 : memref<1x4x16xf32>
  }
}
