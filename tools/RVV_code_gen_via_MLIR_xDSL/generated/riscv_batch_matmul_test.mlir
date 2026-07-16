"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %0 = "emitc_ext.constant"() {value = dense<2.000000e+00> : memref<16x8xf32>} : () -> memref<16x8xf32>
    %alloc = "emitc_ext.constant"() {value = dense<2.000000e+00> : memref<8x16xf32>} : () -> memref<8x16xf32>
    %alloc_1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x8x16xf32>
    %1 = "builtin.unrealized_conversion_cast"(%alloc_1) : (memref<1x8x16xf32>) -> !emitc.ptr<f32>
    %2 = "emitc.call_opaque"(%alloc, %alloc_1) <{callee = "broadcast_operator", args = ["Tensor*", "Tensor*"]}> : (memref<8x16xf32>, memref<1x8x16xf32>) -> memref<1x8x16xf32>
    %alloc_2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x16xf32>
    %3 = "builtin.unrealized_conversion_cast"(%alloc_2) : (memref<1x4x16xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_2) <{operandSegmentSizes = array<i32: 1, 1>}> : (f32, memref<1x4x16xf32>) -> ()
    %4 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %5 = "emitc_ext.constant"() {value = 4 : index} : () -> index
    %6 = "emitc_ext.constant"() {value = 16 : index} : () -> index
    %7 = "emitc_ext.constant"() {value = 8 : index} : () -> index
    %8 = "emitc.call_opaque"(%arg0, %5, %7, %4) <{callee = "create_tensor_A"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %9 = "emitc.call_opaque"(%1, %7, %6, %4) <{callee = "create_tensor_B"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %10 = "emitc.call_opaque"(%3, %5, %6, %4) <{callee = "create_tensor_C"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %11 = "emitc_ext.constant"() {value = 0 : index} : () -> index
    "emitc.call_opaque"(%8, %9, %10, %10, %11, %11) <{callee = "gemm_operator"}> : (!emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, index, index) -> ()
    "rair.batch_matmul"(%arg0, %alloc_1, %alloc_2) : (!emitc.ptr<f32>, memref<1x8x16xf32>, memref<1x4x16xf32>) -> ()
    "func.return"(%alloc_2) : (memref<1x4x16xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "Linear"} : () -> ()
