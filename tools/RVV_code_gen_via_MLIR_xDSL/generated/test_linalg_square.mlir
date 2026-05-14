"builtin.module"() ({
  "func.func"() <{sym_name = "test_square", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %c2_i64 = "emitc_ext.constant"() {value = 2 : i64} : () -> i64
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %1 = "emitc.call_opaque"(%arg0, %alloc) <{callee = "square_operator", args = ["Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<1x128x2048xf32>) -> memref<1x128x2048xf32>
    "linalg.generic"(%arg0, %alloc) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%in : f32, %out : f32):
      %2 = "math.fpowi"(%in, %c2_i64) <{fastmath = #arith.fastmath<none>}> : (f32, i64) -> f32
      "linalg.yield"(%2) : (f32) -> ()
    }) : (!emitc.ptr<f32>, memref<1x128x2048xf32>) -> ()
    "func.return"(%alloc) : (memref<1x128x2048xf32>) -> ()
  }) : () -> ()
}) : () -> ()
