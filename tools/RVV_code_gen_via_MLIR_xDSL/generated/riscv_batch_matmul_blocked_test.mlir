"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %0 = "emitc_ext.constant"() {value = dense<2.000000e+00> : memref<512x2048xf32>} : () -> memref<512x2048xf32>
    %alloc = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2048x512xf32>
    %1 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x2048x512xf32>) -> !emitc.ptr<f32>
    "linalg.generic"(%0, %alloc) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%in : f32, %out : f32):
      "linalg.yield"(%in) : (f32) -> ()
    }) : (memref<512x2048xf32>, memref<1x2048x512xf32>) -> ()
    %alloc_1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x512xf32>
    %2 = "builtin.unrealized_conversion_cast"(%alloc_1) : (memref<1x128x512xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_1) <{operandSegmentSizes = array<i32: 1, 1>}> : (f32, memref<1x128x512xf32>) -> ()
    %3 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %4 = "emitc_ext.constant"() {value = 128 : index} : () -> index
    %5 = "emitc_ext.constant"() {value = 512 : index} : () -> index
    %6 = "emitc_ext.constant"() {value = 2048 : index} : () -> index
    %7 = "emitc.call_opaque"(%arg0, %4, %6, %3) <{callee = "create_tensor_A"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %8 = "emitc.call_opaque"(%1, %6, %5, %3) <{callee = "create_tensor_B"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %9 = "emitc.call_opaque"(%2, %4, %5, %3) <{callee = "create_tensor_C"}> : (!emitc.ptr<f32>, index, index, index) -> !emitc.opaque<"Tensor">
    %10 = "emitc_ext.constant"() {value = 0 : index} : () -> index
    "emitc.call_opaque"(%7, %8, %9, %9, %10, %10) <{callee = "gemm_operator"}> : (!emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, !emitc.opaque<"Tensor">, index, index) -> ()
    "rair.batch_matmul"(%arg0, %alloc, %alloc_1) : (!emitc.ptr<f32>, memref<1x2048x512xf32>, memref<1x128x512xf32>) -> ()
    "func.return"(%alloc_1) : (memref<1x128x512xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "LlamaDecoderBlock"} : () -> ()
