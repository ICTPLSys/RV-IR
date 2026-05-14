"builtin.module"() ({
  "func.func"() <{sym_name = "test_reduce_dim0_sum", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %1 = "emitc.call_opaque"(%arg0, %alloc) <{callee = "reduce_dim0_sum", args = ["Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<1x128x2048xf32>) -> memref<1x128x2048xf32>
    "func.return"(%alloc) : (memref<1x128x2048xf32>) -> ()
  }) : () -> ()
}) : () -> ()
