"builtin.module"() ({
  "func.func"() <{sym_name = "test_elementwise_add", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>, %arg1 : !emitc.ptr<f32>):
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %1 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %2 = "emitc.call_opaque"(%arg0, %arg1, %alloc, %1) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (!emitc.ptr<f32>, !emitc.ptr<f32>, memref<1x128x2048xf32>, i32) -> i32
    "func.return"(%alloc) : (memref<1x128x2048xf32>) -> ()
  }) : () -> ()
}) : () -> ()
