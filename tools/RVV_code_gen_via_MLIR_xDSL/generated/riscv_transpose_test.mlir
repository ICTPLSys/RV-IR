"builtin.module"() ({
  "func.func"() <{sym_name = "transpose_test", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %alloc_input = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x8x64xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc_input) : (memref<1x128x8x64xf32>) -> !emitc.ptr<f32>
    %alloc_output = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x8x128x64xf32>
    %1 = "builtin.unrealized_conversion_cast"(%alloc_output) : (memref<1x8x128x64xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_input) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%2 : f32, %3 : f32):
      "linalg.yield"(%2) : (f32) -> ()
    }) : (f32, memref<1x128x8x64xf32>) -> ()
    %4 = "emitc.call_opaque"(%alloc_input, %alloc_output) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<1x128x8x64xf32>, memref<1x8x128x64xf32>) -> i32
    "func.return"(%alloc_output) : (memref<1x8x128x64xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "TransposeTest"} : () -> ()
