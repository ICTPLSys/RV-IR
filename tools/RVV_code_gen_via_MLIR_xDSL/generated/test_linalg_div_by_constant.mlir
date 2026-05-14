"builtin.module"() ({
  "func.func"() <{sym_name = "test_div_by_constant", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 5.000000e-01 : f32} : () -> f32
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x1xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x128x1xf32>) -> !emitc.ptr<f32>
    %1 = "emitc_ext.constant"() {value = 5.000000e-01 : f32} : () -> f32
    %2 = "emitc_ext.constant"() {value = 8 : i32} : () -> i32
    %3 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %4 = "emitc_ext.constant"() {value = 41 : i32} : () -> i32
    %5 = "emitc.call_opaque"(%arg0, %alloc, %1, %2, %3, %4) <{callee = "tensor_imm_operator", args = ["Tensor*", "Tensor*", "float", "uint32_t", "uint32_t", "uint32_t"]}> : (!emitc.ptr<f32>, memref<1x128x1xf32>, f32, i32, i32, i32) -> i32
    "func.return"(%alloc) : (memref<1x128x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()
