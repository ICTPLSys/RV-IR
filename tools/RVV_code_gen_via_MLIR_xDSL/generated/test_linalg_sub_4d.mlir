"builtin.module"() ({
  "func.func"() <{sym_name = "test_sub_4d_broadcast", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>, %arg1 : !emitc.ptr<f32>):
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32x128x128xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x32x128x128xf32>) -> !emitc.ptr<f32>
    %1 = "emitc.call_opaque"(%arg0, %arg1, %alloc) <{callee = "tensor_tensor_sub", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, memref<1x32x128x128xf32>) -> memref<1x32x128x128xf32>
    "linalg.generic"(%arg0, %arg1, %alloc) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%in : f32, %in_1 : f32, %out : f32):
      %2 = "arith.subf"(%in, %in_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%2) : (f32) -> ()
    }) : (!emitc.ptr<f32>, !emitc.ptr<f32>, memref<1x32x128x128xf32>) -> ()
    "func.return"(%alloc) : (memref<1x32x128x128xf32>) -> ()
  }) : () -> ()
}) : () -> ()
