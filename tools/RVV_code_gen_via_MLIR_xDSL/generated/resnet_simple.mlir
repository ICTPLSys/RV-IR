"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 6.400000e+01 : f32} : () -> f32
    %cst_1 = "emitc_ext.constant"() {value = 1.000000e-05 : f64} : () -> f64
    %cst_2 = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %c1 = "emitc_ext.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %w_conv = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x3x3x3xf32>
    %0 = "builtin.unrealized_conversion_cast"(%w_conv) : (memref<4x3x3x3xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %w_conv) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%1 : f32, %2 : f32):
      "linalg.yield"(%1) : (f32) -> ()
    }) : (f32, memref<4x3x3x3xf32>) -> ()
    %bn_gamma = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %3 = "builtin.unrealized_conversion_cast"(%bn_gamma) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %bn_gamma) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb2(%4 : f32, %5 : f32):
      "linalg.yield"(%4) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %bn_beta = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %6 = "builtin.unrealized_conversion_cast"(%bn_beta) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %bn_beta) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb3(%7 : f32, %8 : f32):
      "linalg.yield"(%7) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %bn_mean = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %9 = "builtin.unrealized_conversion_cast"(%bn_mean) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %bn_mean) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb4(%10 : f32, %11 : f32):
      "linalg.yield"(%10) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %bn_var = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %12 = "builtin.unrealized_conversion_cast"(%bn_var) : (memref<4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %bn_var) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb5(%13 : f32, %14 : f32):
      "linalg.yield"(%13) : (f32) -> ()
    }) : (f32, memref<4xf32>) -> ()
    %fc_w = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x4xf32>
    %15 = "builtin.unrealized_conversion_cast"(%fc_w) : (memref<2x4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %fc_w) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb6(%16 : f32, %17 : f32):
      "linalg.yield"(%16) : (f32) -> ()
    }) : (f32, memref<2x4xf32>) -> ()
    %fc_b = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2xf32>
    %18 = "builtin.unrealized_conversion_cast"(%fc_b) : (memref<2xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %fc_b) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb7(%19 : f32, %20 : f32):
      "linalg.yield"(%19) : (f32) -> ()
    }) : (f32, memref<2xf32>) -> ()
    %alloc = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x3x10x10xf32>
    %21 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<1x3x10x10xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb8(%22 : f32, %23 : f32):
      "linalg.yield"(%22) : (f32) -> ()
    }) : (f32, memref<1x3x10x10xf32>) -> ()
    %subview = "memref.subview"(%alloc) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 3, 4, 4>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (memref<1x3x10x10xf32>) -> memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>
    %24 = "emitc.call_opaque"(%arg0, %subview, %alloc) <{callee = "subview_rowwise_copy_add", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>, memref<1x3x10x10xf32>) -> memref<1x3x4x4xf32, strided<[300, 100, 10, 1], offset: 33>>
    %conv_wrk = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x8x8xf32>
    %25 = "builtin.unrealized_conversion_cast"(%conv_wrk) : (memref<1x4x8x8xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %conv_wrk) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb9(%26 : f32, %27 : f32):
      "linalg.yield"(%26) : (f32) -> ()
    }) : (f32, memref<1x4x8x8xf32>) -> ()
    %28 = "emitc.call_opaque"(%alloc, %w_conv, %conv_wrk) <{callee = "conv_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x3x10x10xf32>, memref<4x3x3x3xf32>, memref<1x4x8x8xf32>) -> i32
    %bn_out = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x8x8xf32>
    %29 = "builtin.unrealized_conversion_cast"(%bn_out) : (memref<1x4x8x8xf32>) -> !emitc.ptr<f32>
    %30 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %31 = "emitc.call_opaque"(%conv_wrk, %bn_mean, %bn_out, %30) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x8x8xf32>, memref<4xf32>, memref<1x4x8x8xf32>, i32) -> i32
    %32 = "emitc_ext.constant"() {value = 40 : i32} : () -> i32
    %33 = "emitc.call_opaque"(%bn_out, %bn_gamma, %bn_out, %32) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x8x8xf32>, memref<4xf32>, memref<1x4x8x8xf32>, i32) -> i32
    %34 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %35 = "emitc.call_opaque"(%bn_out, %bn_beta, %bn_out, %34) <{callee = "tensor_vector_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x4x8x8xf32>, memref<4xf32>, memref<1x4x8x8xf32>, i32) -> i32
    %relu_out = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x8x8xf32>
    %36 = "builtin.unrealized_conversion_cast"(%relu_out) : (memref<1x4x8x8xf32>) -> !emitc.ptr<f32>
    %37 = "emitc.call_opaque"(%bn_out, %relu_out) <{callee = "relu_operator", args = ["Tensor*", "Tensor*"]}> : (memref<1x4x8x8xf32>, memref<1x4x8x8xf32>) -> i32
    %pool_k = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x8xf32>
    %38 = "builtin.unrealized_conversion_cast"(%pool_k) : (memref<8x8xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1, %pool_k) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb10(%39 : f32, %40 : f32):
      "linalg.yield"(%39) : (f32) -> ()
    }) : (f32, memref<8x8xf32>) -> ()
    %pooled_wrk = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x1x1xf32>
    %41 = "builtin.unrealized_conversion_cast"(%pooled_wrk) : (memref<1x4x1x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %pooled_wrk) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb11(%42 : f32, %43 : f32):
      "linalg.yield"(%42) : (f32) -> ()
    }) : (f32, memref<1x4x1x1xf32>) -> ()
    %44 = "emitc.call_opaque"(%relu_out, %pool_k, %pooled_wrk) <{callee = "pooling_nchw_sum", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x4x8x8xf32>, memref<8x8xf32>, memref<1x4x1x1xf32>) -> i32
    %pooled_avg = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x1x1xf32>
    %45 = "builtin.unrealized_conversion_cast"(%pooled_avg) : (memref<1x4x1x1xf32>) -> !emitc.ptr<f32>
    %46 = "emitc_ext.constant"() {value = 6.400000e+01 : f32} : () -> f32
    %47 = "emitc_ext.constant"() {value = 8 : i32} : () -> i32
    %48 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %49 = "emitc_ext.constant"() {value = 41 : i32} : () -> i32
    %50 = "emitc.call_opaque"(%pooled_wrk, %pooled_avg, %46, %47, %48, %49) <{callee = "tensor_imm_operator", args = ["Tensor*", "Tensor*", "float", "uint32_t", "uint32_t", "uint32_t"]}> : (memref<1x4x1x1xf32>, memref<1x4x1x1xf32>, f32, i32, i32, i32) -> i32
    %collapsed = "emitc.call_opaque"(%pooled_avg) <{callee = "flatten_view_operator", args = ["Tensor*"]}> : (memref<1x4x1x1xf32>) -> memref<1x4xf32>
    %fc_w_t = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x2xf32>
    %51 = "builtin.unrealized_conversion_cast"(%fc_w_t) : (memref<4x2xf32>) -> !emitc.ptr<f32>
    %52 = "emitc.call_opaque"(%fc_w, %fc_w_t) <{callee = "transpose_operator", args = ["Tensor*", "Tensor*"]}> : (memref<2x4xf32>, memref<4x2xf32>) -> i32
    %logits = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2xf32>
    %53 = "builtin.unrealized_conversion_cast"(%logits) : (memref<1x2xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %logits) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb12(%54 : f32, %55 : f32):
      "linalg.yield"(%54) : (f32) -> ()
    }) : (f32, memref<1x2xf32>) -> ()
    %56 = "emitc.call_opaque"(%collapsed, %fc_w_t, %logits) <{callee = "matmul_operator", args = ["Tensor*", "Tensor*", "Tensor*"]}> : (memref<1x4xf32>, memref<4x2xf32>, memref<1x2xf32>) -> i32
    %out = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x2xf32>
    %57 = "builtin.unrealized_conversion_cast"(%out) : (memref<1x2xf32>) -> !emitc.ptr<f32>
    %58 = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %59 = "emitc.call_opaque"(%logits, %fc_b, %out, %58) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<1x2xf32>, memref<2xf32>, memref<1x2xf32>, i32) -> i32
    "func.return"(%out) : (memref<1x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = () -> i32}> ({
    %c0 = "emitc_ext.constant"() {value = 0 : index} : () -> index
    %c1 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %c0i = "emitc_ext.constant"() {value = 0 : i32} : () -> i32
    %c1i = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %c52 = "emitc_ext.constant"() {value = 52 : i32} : () -> i32
    %c1f = "emitc_ext.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %in = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x3x4x4xf32>
    %0 = "builtin.unrealized_conversion_cast"(%in) : (memref<1x3x4x4xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%c1f, %in) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1 : f32, %2 : f32):
      "linalg.yield"(%1) : (f32) -> ()
    }) : (f32, memref<1x3x4x4xf32>) -> ()
    %res = "func.call"(%in) <{callee = @forward}> : (memref<1x3x4x4xf32>) -> memref<1x2xf32>
    %v0 = "memref.load"(%res, %c0, %c0) : (memref<1x2xf32>, index, index) -> f32
    %v1 = "memref.load"(%res, %c0, %c1) : (memref<1x2xf32>, index, index) -> f32
    %i0 = "arith.fptosi"(%v0) : (f32) -> i32
    %i1 = "arith.fptosi"(%v1) : (f32) -> i32
    %sum = "emitc.add"(%i0, %i1) : (i32, i32) -> i32
    %ok = "arith.cmpi"(%sum, %c52) <{predicate = 0 : i64}> : (i32, i32) -> i1
    %ret = "arith.select"(%ok, %c0i, %c1i) : (i1, i32, i32) -> i32
    "func.return"(%ret) : (i32) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "ResNetSimple"} : () -> ()