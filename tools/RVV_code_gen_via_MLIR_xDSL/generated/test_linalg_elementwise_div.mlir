"builtin.module"() ({
  "func.func"() <{sym_name = "test_elementwise_div", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>, %arg1 : !emitc.ptr<f32>):
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %cst = "emitc_ext.constant"() {value = 2.048000e+03 : f32} : () -> f32
    %1 = "emitc.call_opaque"(%arg0, %arg1, %alloc) <{callee = "div_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, memref<1x128x2048xf32>) -> i32
    "func.return"(%alloc) : (memref<1x128x2048xf32>) -> ()
  }) : () -> ()
}) : () -> ()
