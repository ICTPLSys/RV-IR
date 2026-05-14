"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %c0_i64 = "emitc_ext.constant"() {value = 0 : i64} : () -> i64
    %cst_1 = "emitc_ext.constant"() {value = 0xff800000 : f32} : () -> f32
    %c1 = "emitc_ext.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %0 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32xf32>
    %1 = "builtin.unrealized_conversion_cast"(%0) : (memref<32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%2 : f32, %3 : f32):
      "linalg.yield"(%2) : (f32) -> ()
    }) : (f32, memref<32xf32>) -> ()
    %4 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x196xf32>
    %5 = "builtin.unrealized_conversion_cast"(%4) : (memref<32x196xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%6 : f32, %7 : f32):
      "linalg.yield"(%6) : (f32) -> ()
    }) : (f32, memref<32x196xf32>) -> ()
    %8 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
    %9 = "builtin.unrealized_conversion_cast"(%8) : (memref<10xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %8) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb3(%10 : f32, %11 : f32):
      "linalg.yield"(%10) : (f32) -> ()
    }) : (f32, memref<10xf32>) -> ()
    %12 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x32xf32>
    %13 = "builtin.unrealized_conversion_cast"(%12) : (memref<10x32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %12) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb4(%14 : f32, %15 : f32):
      "linalg.yield"(%14) : (f32) -> ()
    }) : (f32, memref<10x32xf32>) -> ()
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x14x14xf32>
    %16 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x1x14x14xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_1, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb5(%17 : f32, %18 : f32):
      "linalg.yield"(%17) : (f32) -> ()
    }) : (f32, memref<1x1x14x14xf32>) -> ()
    %alloc_1 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x2xf32>
    %19 = "builtin.unrealized_conversion_cast"(%alloc_1) : (memref<2x2xf32>) -> !emitc.ptr<f32>
    %alloc_2 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x14x14xf32>
    %20 = "builtin.unrealized_conversion_cast"(%alloc_2) : (memref<1x1x14x14xf32>) -> !emitc.ptr<f32>
    "linalg.pooling_nchw_max"(%arg0, %alloc_1, %alloc_2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb6(%21 : f32, %22 : f32, %23 : f32):
      %24 = "arith.maximumf"(%21, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%24) : (f32) -> ()
    }) {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} : (!emitc.ptr<f32>, memref<2x2xf32>, memref<1x1x14x14xf32>) -> ()
    %collapse_shape = "emitc.call_opaque"(%alloc_2) <{callee = "flatten_view_operator", args = ["Tensor*"]}> : (memref<1x1x14x14xf32>) -> memref<1x196xf32>
    %alloc_3 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<196x32xf32>
    %25 = "builtin.unrealized_conversion_cast"(%alloc_3) : (memref<196x32xf32>) -> !emitc.ptr<f32>
    %26 = "emitc.call_opaque"(%4, %alloc_3) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<32x196xf32>, memref<196x32xf32>) -> i32
    %alloc_4 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %27 = "builtin.unrealized_conversion_cast"(%alloc_4) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_4) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb7(%28 : f32, %29 : f32):
      "linalg.yield"(%28) : (f32) -> ()
    }) : (f32, memref<1x32xf32>) -> ()
    %alloc_5 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %30 = "builtin.unrealized_conversion_cast"(%alloc_5) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %31 = "emitc.call_opaque"(%collapse_shape, %alloc_3, %alloc_5) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x196xf32>, memref<196x32xf32>, memref<1x32xf32>) -> i32
    %alloc_6 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %32 = "builtin.unrealized_conversion_cast"(%alloc_6) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %33 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %34 = "emitc.call_opaque"(%alloc_5, %0, %alloc_6, %33) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x32xf32>, memref<32xf32>, memref<1x32xf32>, i32) -> i32
    %alloc_7 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32xf32>
    %35 = "builtin.unrealized_conversion_cast"(%alloc_7) : (memref<1x32xf32>) -> !emitc.ptr<f32>
    %36 = "emitc.call_opaque"(%alloc_6, %alloc_7) <{callee = "relu_operator", args = ["Tensor*", "Tensor*"]}> : (memref<1x32xf32>, memref<1x32xf32>) -> i32
    %alloc_8 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x10xf32>
    %37 = "builtin.unrealized_conversion_cast"(%alloc_8) : (memref<32x10xf32>) -> !emitc.ptr<f32>
    %38 = "emitc.call_opaque"(%12, %alloc_8) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<10x32xf32>, memref<32x10xf32>) -> i32
    %alloc_9 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %39 = "builtin.unrealized_conversion_cast"(%alloc_9) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_9) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb8(%40 : f32, %41 : f32):
      "linalg.yield"(%40) : (f32) -> ()
    }) : (f32, memref<1x10xf32>) -> ()
    %alloc_10 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %42 = "builtin.unrealized_conversion_cast"(%alloc_10) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %43 = "emitc.call_opaque"(%alloc_7, %alloc_8, %alloc_10) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x32xf32>, memref<32x10xf32>, memref<1x10xf32>) -> i32
    %alloc_11 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %44 = "builtin.unrealized_conversion_cast"(%alloc_11) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %45 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %46 = "emitc.call_opaque"(%alloc_10, %8, %alloc_11, %45) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x10xf32>, memref<10xf32>, memref<1x10xf32>, i32) -> i32
    %alloc_12 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi64>
    %47 = "builtin.unrealized_conversion_cast"(%alloc_12) : (memref<1xi64>) -> !emitc.ptr<i64>
    "linalg.fill"(%c0_i64, %alloc_12) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb9(%48 : i64, %49 : i64):
      "linalg.yield"(%48) : (i64) -> ()
    }) : (i64, memref<1xi64>) -> ()
    %alloc_13 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    %50 = "builtin.unrealized_conversion_cast"(%alloc_13) : (memref<1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_1, %alloc_13) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb10(%51 : f32, %52 : f32):
      "linalg.yield"(%51) : (f32) -> ()
    }) : (f32, memref<1xf32>) -> ()
    %alloc_14 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    %53 = "builtin.unrealized_conversion_cast"(%alloc_14) : (memref<1xf32>) -> !emitc.ptr<f32>
    %alloc_15 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi64>
    %54 = "builtin.unrealized_conversion_cast"(%alloc_15) : (memref<1xi64>) -> !emitc.ptr<i64>
    %55 = "emitc.call_opaque"(%alloc_11, %alloc_14) <{callee = "reduce_dim1_max", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1xf32>) -> i32
    %expand_shape = "memref.expand_shape"(%alloc_14) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1>}> : (memref<1xf32>) -> memref<1x1xf32>
    %alloc_16 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %56 = "builtin.unrealized_conversion_cast"(%alloc_16) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %57 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %58 = "emitc.call_opaque"(%alloc_11, %expand_shape, %alloc_16, %57) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x10xf32>, memref<1x1xf32>, memref<1x10xf32>, i32) -> i32
    %alloc_17 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %59 = "builtin.unrealized_conversion_cast"(%alloc_17) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %60 = "emitc.call_opaque"(%alloc_16, %alloc_17) <{callee = "lut_exp", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x10xf32>) -> memref<1x10xf32>
    %alloc_18 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf32>
    %61 = "builtin.unrealized_conversion_cast"(%alloc_18) : (memref<1x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc_18) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb11(%62 : f32, %63 : f32):
      "linalg.yield"(%62) : (f32) -> ()
    }) : (f32, memref<1x1xf32>) -> ()
    %alloc_19 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf32>
    %64 = "builtin.unrealized_conversion_cast"(%alloc_19) : (memref<1x1xf32>) -> !emitc.ptr<f32>
    %65 = "emitc.call_opaque"(%alloc_17, %alloc_19) <{callee = "reduce_dim0_sum", args = ["Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x1xf32>) -> memref<1x1xf32>
    %alloc_20 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
    %66 = "builtin.unrealized_conversion_cast"(%alloc_20) : (memref<1x10xf32>) -> !emitc.ptr<f32>
    %67 = "emitc.call_opaque"(%alloc_17, %alloc_19, %alloc_20) <{callee = "div_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x10xf32>, memref<1x1xf32>, memref<1x10xf32>) -> i32
    "func.return"(%alloc_20) : (memref<1x10xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "MnistNet"} : () -> ()
