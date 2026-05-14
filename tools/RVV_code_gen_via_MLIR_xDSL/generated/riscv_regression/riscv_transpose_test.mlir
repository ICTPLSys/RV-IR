"builtin.module"() ({
  "func.func"() <{sym_name = "transpose_test", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %alloc_input = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x8x64xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc_input) : (memref<1x128x8x64xf32>) -> !emitc.ptr<f32>
    %alloc_output = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x8x128x64xf32>
    %1 = "builtin.unrealized_conversion_cast"(%alloc_output) : (memref<1x8x128x64xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_input) <{operandSegmentSizes = array<i32: 1, 1>}> : (f32, memref<1x128x8x64xf32>) -> ()
    %2 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %3 = "emitc_ext.constant"() {value = 128 : index} : () -> index
    %4 = "emitc_ext.constant"() {value = 8 : index} : () -> index
    %5 = "emitc_ext.constant"() {value = 64 : index} : () -> index
    %6 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %7 = "emitc_ext.constant"() {value = 8 : index} : () -> index
    %8 = "emitc_ext.constant"() {value = 128 : index} : () -> index
    %9 = "emitc_ext.constant"() {value = 64 : index} : () -> index
    %10 = "emitc.call_opaque"(%0, %2, %3, %4, %5) <{callee = "create_tensor_transpose_in"}> : (!emitc.ptr<f32>, index, index, index, index) -> !emitc.opaque<"Tensor">
    %11 = "emitc.call_opaque"(%1, %6, %7, %8, %9) <{callee = "create_tensor_transpose_out"}> : (!emitc.ptr<f32>, index, index, index, index) -> !emitc.opaque<"Tensor">
    %12 = "emitc_ext.constant"() {value = 1 : i64} : () -> i64
    "emitc.call_opaque"(%10, %11, %12) <{callee = "transpose_operator"}> : (!emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, i64) -> ()
    "func.return"(%alloc_output) : (memref<1x8x128x64xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "TransposeTest"} : () -> ()
