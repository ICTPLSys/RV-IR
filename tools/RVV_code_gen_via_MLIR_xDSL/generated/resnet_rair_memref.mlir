"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0: !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 8.100000e+01 : f32} : () -> f32
    %cst_1 = "emitc_ext.constant"() {value = 1.000000e-05 : f64} : () -> f64
    %cst_2 = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %cst_3 = "emitc_ext.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%1: f32, %2: f32):
      "linalg.yield"(%1) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %alloc_1 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x3x3x3xf32>
    %3 = "builtin.unrealized_conversion_cast"(%alloc_1) : (memref<4x3x3x3xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc_1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%4: f32, %5: f32):
      "linalg.yield"(%4) : (f32) -> ()
    }) : (f32, memref<4x3x3x3xf32>) -> ()
    %alloc_2 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %6 = "builtin.unrealized_conversion_cast"(%alloc_2) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc_2) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb3(%7: f32, %8: f32):
      "linalg.yield"(%7) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %alloc_3 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %9 = "builtin.unrealized_conversion_cast"(%alloc_3) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc_3) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb4(%10: f32, %11: f32):
      "linalg.yield"(%10) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %alloc_4 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %12 = "builtin.unrealized_conversion_cast"(%alloc_4) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc_4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb5(%13: f32, %14: f32):
      "linalg.yield"(%13) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %alloc_5 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xf32>
    %15 = "builtin.unrealized_conversion_cast"(%alloc_5) : (memref<4x4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_3, %alloc_5) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb6(%16: f32, %17: f32):
      "linalg.yield"(%16) : (f32) -> ()
    }) : (f32, memref<4x4xf32>) -> ()
    %alloc_6 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x3x11x11xf32>
    %18 = "builtin.unrealized_conversion_cast"(%alloc_6) : (memref<1x3x11x11xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_6) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb7(%19: f32, %20: f32):
      "linalg.yield"(%19) : (f32) -> ()
    }) : (f32, memref<1x3x11x11xf32>) -> ()
    %subview = "memref.subview"(%alloc_6) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 3, 5, 5>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (memref<1x3x11x11xf32>) -> memref<1x3x5x5xf32, strided<[363, 121, 11, 1], offset: 36>>
    %alloc_7 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x9x9xf32>
    %21 = "builtin.unrealized_conversion_cast"(%alloc_7) : (memref<1x4x9x9xf32>) -> !emitc.ptr<f32>
    "linalg.broadcast"(%alloc, %alloc_7) <{dimensions = array<i64: 0, 2, 3>}> ({
    ^bb8(%22: f32, %23: f32):
      "linalg.yield"(%22) : (f32) -> ()
    }) : (memref<4xf32>, memref<1x4x9x9xf32>) -> ()
    %alloc_8 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x9x9xf32>
    %24 = "builtin.unrealized_conversion_cast"(%alloc_8) : (memref<1x4x9x9xf32>) -> !emitc.ptr<f32>
    %25 = "emitc.call_opaque"(%alloc_6, %alloc_1, %alloc_8) <{callee = "conv_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x3x11x11xf32>, memref<4x3x3x3xf32>, memref<1x4x9x9xf32>) -> i32
    %alloc_9 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x9x9xf32>
    %26 = "builtin.unrealized_conversion_cast"(%alloc_9) : (memref<1x4x9x9xf32>) -> !emitc.ptr<f32>
    %27 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %28 = "emitc.call_opaque"(%alloc_8, %alloc_3, %alloc_9, %27) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x9x9xf32>, memref<4xf32>, memref<1x4x9x9xf32>, i32) -> i32
    %29 = "emitc_ext.constant"() {value = 40 : i32} : () -> i32
    %30 = "emitc.call_opaque"(%alloc_9, %alloc_2, %alloc_9, %29) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x9x9xf32>, memref<4xf32>, memref<1x4x9x9xf32>, i32) -> i32
    %31 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %32 = "emitc.call_opaque"(%alloc_9, %alloc_3, %alloc_9, %31) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x9x9xf32>, memref<4xf32>, memref<1x4x9x9xf32>, i32) -> i32
    %alloc_10 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x9x9xf32>
    %33 = "builtin.unrealized_conversion_cast"(%alloc_10) : (memref<1x4x9x9xf32>) -> !emitc.ptr<f32>
    %34 = "emitc.call_opaque"(%alloc_9, %alloc_10) <{callee = "relu_operator", args = ["Tensor*", "Tensor*"]}> : (memref<1x4x9x9xf32>, memref<1x4x9x9xf32>) -> i32
    %alloc_11 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x1x1xf32>
    %35 = "builtin.unrealized_conversion_cast"(%alloc_11) : (memref<1x4x1x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_11) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb9(%36: f32, %37: f32):
      "linalg.yield"(%36) : (f32) -> ()
    }) : (f32, memref<1x4x1x1xf32>) -> ()
    %alloc_12 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<9x9xf32>
    %38 = "builtin.unrealized_conversion_cast"(%alloc_12) : (memref<9x9xf32>) -> !emitc.ptr<f32>
    %alloc_13 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x1x1xf32>
    %39 = "builtin.unrealized_conversion_cast"(%alloc_13) : (memref<1x4x1x1xf32>) -> !emitc.ptr<f32>
    %40 = "emitc.call_opaque"(%alloc_10, %alloc_12, %alloc_13) <{callee = "pooling_nchw_sum", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x4x9x9xf32>, memref<9x9xf32>, memref<1x4x1x1xf32>) -> i32
    %alloc_14 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x1x1xf32>
    %41 = "builtin.unrealized_conversion_cast"(%alloc_14) : (memref<1x4x1x1xf32>) -> !emitc.ptr<f32>
    %42 = "emitc_ext.constant"() {value = 8.100000e+01 : f32} : () -> f32
    %43 = "emitc_ext.constant"() {value = 8 : i32} : () -> i32
    %44 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %45 = "emitc_ext.constant"() {value = 41 : i32} : () -> i32
    %46 = "emitc.call_opaque"(%alloc_13, %alloc_14, %42, %43, %44, %45) <{callee = "tensor_imm_operator", args = ["Tensor*", "Tensor*", "float", "uint32_t", "uint32_t", "uint32_t"]}> : (memref<1x4x1x1xf32>, memref<1x4x1x1xf32>, f32, i32, i32, i32) -> i32
    %collapse_shape = "emitc.call_opaque"(%alloc_14) <{callee = "flatten_view_operator", args = ["Tensor*"]}> : (memref<1x4x1x1xf32>) -> memref<1x4xf32>
    %alloc_15 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4xf32>
    %47 = "builtin.unrealized_conversion_cast"(%alloc_15) : (memref<1x4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_15) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb10(%48: f32, %49: f32):
      "linalg.yield"(%48) : (f32) -> ()
    }) : (f32, memref<1x4xf32>) -> ()
    %alloc_16 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4xf32>
    %50 = "builtin.unrealized_conversion_cast"(%alloc_16) : (memref<1x4xf32>) -> !emitc.ptr<f32>
    %51 = "emitc.call_opaque"(%collapse_shape, %alloc_5, %alloc_16) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x4xf32>, memref<4x4xf32>, memref<1x4xf32>) -> i32
    %alloc_17 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4xf32>
    %52 = "builtin.unrealized_conversion_cast"(%alloc_17) : (memref<1x4xf32>) -> !emitc.ptr<f32>
    %53 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %54 = "emitc.call_opaque"(%alloc_16, %alloc_4, %alloc_17, %53) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4xf32>, memref<4xf32>, memref<1x4xf32>, i32) -> i32
    "func.return"(%alloc_17) : (memref<1x4xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "ResNetSimple"} : () -> ()