"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %c0_i64 = "emitc_ext.constant"() {value = 0 : i64} : () -> i64
    %cst_1 = "emitc_ext.constant"() {value = 0xff800000 : f32} : () -> f32
    %cst_2 = "emitc_ext.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%1 : f32, %2 : f32):
      "linalg.yield"(%1) : (f32) -> ()
    }) : (f32, memref<32xf32>) -> ()
    %alloc_1 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x196xf32>
    %3 = "builtin.unrealized_conversion_cast"(%alloc_1) : (memref<32x196xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_1) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%4 : f32, %5 : f32):
      "linalg.yield"(%4) : (f32) -> ()
    }) : (f32, memref<32x196xf32>) -> ()
    %alloc_2 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
    %6 = "builtin.unrealized_conversion_cast"(%alloc_2) : (memref<10xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_2) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb3(%7 : f32, %8 : f32):
      "linalg.yield"(%7) : (f32) -> ()
    }) : (f32, memref<10xf32>) -> ()
    %alloc_3 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x32xf32>
    %9 = "builtin.unrealized_conversion_cast"(%alloc_3) : (memref<10x32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_3) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb4(%10 : f32, %11 : f32):
      "linalg.yield"(%10) : (f32) -> ()
    }) : (f32, memref<10x32xf32>) -> ()
    %alloc_4 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x14x14xf32>
    %12 = "builtin.unrealized_conversion_cast"(%alloc_4) : (memref<1x1x14x14xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_1, %alloc_4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb5(%13 : f32, %14 : f32):
      "linalg.yield"(%13) : (f32) -> ()
    }) : (f32, memref<1x1x14x14xf32>) -> ()
    %alloc_5 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x2xf32>
    %15 = "builtin.unrealized_conversion_cast"(%alloc_5) : (memref<2x2xf32>) -> !emitc.ptr<f32>
    %alloc_6 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x14x14xf32>
    %16 = "builtin.unrealized_conversion_cast"(%alloc_6) : (memref<1x1x14x14xf32>) -> !emitc.ptr<f32>
    "linalg.pooling_nchw_max"(%arg0, %alloc_5, %alloc_6) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb6(%17 : f32, %18 : f32, %19 : f32):
      %20 = "arith.maximumf"(%17, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%20) : (f32) -> ()
    }) {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} : (!emitc.ptr<f32>, memref<2x2xf32>, memref<1x1x14x14xf32>) -> ()
    %collapse_shape = "emitc.call_opaque"(%alloc_6) <{callee = "flatten_view_operator", args = ["Tensor*"]}> : (memref<1x1x14x14xf32>) -> memref<1x196xf32>
    %alloc_7 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<196x32xf32>
    %21 = "builtin.unrealized_conversion_cast"(%alloc_7) : (memref<196x32xf32>) -> !emitc.ptr<f32>
    %22 = "emitc.call_opaque"(%alloc_1, %alloc_7) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<32x196xf32>, memref<196x32xf32>) -> i32
    %alloc_8 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %23 = "builtin.unrealized_conversion_cast"(%alloc_8) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_8) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb7(%24 : f32, %25 : f32):
      "linalg.yield"(%24) : (f32) -> ()
    }) : (f32, memref<1x32xf32>) -> ()
    %alloc_9 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %26 = "builtin.unrealized_conversion_cast"(%alloc_9) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %27 = "emitc.call_opaque"(%collapse_shape, %alloc_7, %alloc_9) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x196xf32>, memref<196x32xf32>, memref<1x32xf32>) -> i32
    %alloc_10 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %28 = "builtin.unrealized_conversion_cast"(%alloc_10) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %29 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %30 = "emitc.call_opaque"(%alloc_9, %alloc, %alloc_10, %29) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x32xf32>, memref<32xf32>, memref<1x32xf32>, i32) -> i32
    %alloc_11 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %31 = "builtin.unrealized_conversion_cast"(%alloc_11) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %32 = "emitc.call_opaque"(%alloc_10, %alloc_11) <{callee = "relu_operator", args = ["Tensor*", "Tensor*"]}> : (memref<1x32xf32>, memref<1x32xf32>) -> i32
    %alloc_12 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x10xf32>
    %33 = "builtin.unrealized_conversion_cast"(%alloc_12) : (memref<32x10xf32>) -> !emitc.ptr<f32>
    %34 = "emitc.call_opaque"(%alloc_3, %alloc_12) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<10x32xf32>, memref<32x10xf32>) -> i32
    %alloc_13 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %35 = "builtin.unrealized_conversion_cast"(%alloc_13) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_13) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb8(%36 : f32, %37 : f32):
      "linalg.yield"(%36) : (f32) -> ()
    }) : (f32, memref<1x10xf32>) -> ()
    %alloc_14 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %38 = "builtin.unrealized_conversion_cast"(%alloc_14) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %39 = "emitc.call_opaque"(%alloc_11, %alloc_12, %alloc_14) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x32xf32>, memref<32x10xf32>, memref<1x10xf32>) -> i32
    %alloc_15 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %40 = "builtin.unrealized_conversion_cast"(%alloc_15) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %41 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %42 = "emitc.call_opaque"(%alloc_14, %alloc_2, %alloc_15, %41) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x10xf32>, memref<10xf32>, memref<1x10xf32>, i32) -> i32
    %alloc_16 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi64>
    %43 = "builtin.unrealized_conversion_cast"(%alloc_16) : (memref<1xi64>) -> !emitc.ptr<i64>
    "linalg.fill"(%c0_i64, %alloc_16) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb9(%44 : i64, %45 : i64):
      "linalg.yield"(%44) : (i64) -> ()
    }) : (i64, memref<1xi64>) -> ()
    %alloc_17 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    %46 = "builtin.unrealized_conversion_cast"(%alloc_17) : (memref<1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_1, %alloc_17) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb10(%47 : f32, %48 : f32):
      "linalg.yield"(%47) : (f32) -> ()
    }) : (f32, memref<1xf32>) -> ()
    %alloc_18 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    %49 = "builtin.unrealized_conversion_cast"(%alloc_18) : (memref<1xf32>) -> !emitc.ptr<f32>
    %alloc_19 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi64>
    %50 = "builtin.unrealized_conversion_cast"(%alloc_19) : (memref<1xi64>) -> !emitc.ptr<i64>
    %51 = "emitc.call_opaque"(%alloc_15, %alloc_18) <{callee = "reduce_dim1_max", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1xf32>) -> i32
    %expand_shape = "memref.expand_shape"(%alloc_18) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1>}> : (memref<1xf32>) -> memref<1x1xf32>
    %alloc_20 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %52 = "builtin.unrealized_conversion_cast"(%alloc_20) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %53 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %54 = "emitc.call_opaque"(%alloc_15, %expand_shape, %alloc_20, %53) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x10xf32>, memref<1x1xf32>, memref<1x10xf32>, i32) -> i32
    %alloc_21 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %55 = "builtin.unrealized_conversion_cast"(%alloc_21) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %56 = "emitc.call_opaque"(%alloc_20, %alloc_21) <{callee = "lut_exp", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x10xf32>) -> memref<1x10xf32>
    %alloc_22 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf32>
    %57 = "builtin.unrealized_conversion_cast"(%alloc_22) : (memref<1x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_22) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb11(%58 : f32, %59 : f32):
      "linalg.yield"(%58) : (f32) -> ()
    }) : (f32, memref<1x1xf32>) -> ()
    %alloc_23 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf32>
    %60 = "builtin.unrealized_conversion_cast"(%alloc_23) : (memref<1x1xf32>) -> !emitc.ptr<f32>
    %61 = "emitc.call_opaque"(%alloc_21, %alloc_23) <{callee = "reduce_dim0_sum", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x1xf32>) -> memref<1x1xf32>
    %alloc_24 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %62 = "builtin.unrealized_conversion_cast"(%alloc_24) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %63 = "emitc.call_opaque"(%alloc_21, %alloc_23, %alloc_24) <{callee = "div_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x1xf32>, memref<1x10xf32>) -> i32
    "func.return"(%alloc_24) : (memref<1x10xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "MnistNet"} : () -> ()
