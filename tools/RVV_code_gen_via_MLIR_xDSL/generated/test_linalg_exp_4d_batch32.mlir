"builtin.module"() ({
  "func.func"() <{sym_name = "test_exp_4d_batch32", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32x128x128xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<32x32x128x128xf32>) -> !emitc.ptr<f32>
    %1 = "emitc.call_opaque"(%arg0, %alloc) <{callee = "lut_exp", args = ["Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<32x32x128x128xf32>) -> memref<32x32x128x128xf32>
    "func.return"(%alloc) : (memref<32x32x128x128xf32>) -> ()
  }) : () -> ()
}) : () -> ()
