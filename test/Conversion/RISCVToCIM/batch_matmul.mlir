// RUN: torch-mlir-opt <%s --convert-rair-to-cim | FileCheck %s

// CHECK: call @llvm.riscv.trans.drv(%{{.*}}) : (i32) -> i32    
// CHECK: call @llvm.riscv.conv.drv(%{{.*}}, %{{.*}}) : (i32, i32) -> i32  
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "Linear"} {
  memref.global "private" constant @__constant_16x8xf32 : memref<16x8xf32> = dense<2.0> 
  func.func @forward(%arg0: memref<1x4x8xf32>) -> memref<1x4x16xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = memref.get_global @__constant_16x8xf32 : memref<16x8xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    rair.transpose ins(%0 : memref<16x8xf32>) outs(%alloc : memref<8x16xf32>) {permutation = array<i64: 1, 0>}
    %alloc_0 = memref.alloc() : memref<1x8x16xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<8x16xf32>) outs(%alloc_0 : memref<1x8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_1 = memref.alloc() : memref<1x4x16xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<1x4x16xf32>)
    rair.batch_matmul ins(%arg0, %alloc_0 : memref<1x4x8xf32>, memref<1x8x16xf32>) outs(%alloc_1 : memref<1x4x16xf32>)
    return %alloc_1 : memref<1x4x16xf32>
  }
}