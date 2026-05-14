"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %alloc = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32x1xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x32x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%1 : f32, %2 : f32):
      "linalg.yield"(%1) : (f32) -> ()
    }) : (f32, memref<1x32x1xf32>) -> ()
    %3 = "emitc.call_opaque"(%arg0, %alloc) <{callee = "reduce_dim2_sum", args = ["Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<1x32x1xf32>) -> memref<1x32x1xf32>
    "linalg.generic"(%arg0, %alloc) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%in : f32, %out : f32):
      %4 = "emitc.add"(%in, %out) : (f32, f32) -> f32
      "linalg.yield"(%4) : (f32) -> ()
    }) : (!emitc.ptr<f32>, memref<1x32x1xf32>) -> ()
    "func.return"(%alloc) : (memref<1x32x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()
