"builtin.module"() ({
  "func.func"() <{sym_name = "rmsnorm_test", function_type = (!emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0 : !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 2.048000e+03 : f32} : () -> f32
    %cst_1 = "emitc_ext.constant"() {value = dense<1.000000e-05> : memref<f64>} : () -> memref<f64>
    %c2_i64 = "emitc_ext.constant"() {value = 2 : i64} : () -> i64
    %cst_2 = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %gamma = "emitc_ext.constant"() {value = dense<1.000000e+00> : memref<2048xf32>} : () -> memref<2048xf32>
    %alloc_square = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc_square) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %1 = "emitc.call_opaque"(%arg0, %alloc_square) <{callee = "square_operator", args = ["Tensor*", "Tensor*"]}> : (!emitc.ptr<f32>, memref<1x128x2048xf32>) -> memref<1x128x2048xf32>
    %alloc_sum = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x1xf32>
    %2 = "builtin.unrealized_conversion_cast"(%alloc_sum) : (memref<1x128x1xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst_2, %alloc_sum) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%3 : f32, %4 : f32):
      "linalg.yield"(%3) : (f32) -> ()
    }) : (f32, memref<1x128x1xf32>) -> ()
    %5 = "emitc.call_opaque"(%alloc_square, %alloc_sum) <{callee = "reduce_dim0_sum", args = ["Tensor*", "Tensor*"]}> : (memref<1x128x2048xf32>, memref<1x128x1xf32>) -> memref<1x128x1xf32>
    %alloc_mean = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x1xf32>
    %6 = "builtin.unrealized_conversion_cast"(%alloc_mean) : (memref<1x128x1xf32>) -> !emitc.ptr<f32>
    %7 = "emitc_ext.constant"() {value = 2.048000e+03 : f32} : () -> f32
    %8 = "emitc_ext.constant"() {value = 8 : i32} : () -> i32
    %9 = "emitc_ext.constant"() {value = 1 : i32} : () -> i32
    %10 = "emitc_ext.constant"() {value = 41 : i32} : () -> i32
    %11 = "emitc.call_opaque"(%alloc_sum, %alloc_mean, %7, %8, %9, %10) <{callee = "tensor_imm_operator", args = ["Tensor*", "Tensor*", "float", "uint32_t", "uint32_t", "uint32_t"]}> : (memref<1x128x1xf32>, memref<1x128x1xf32>, f32, i32, i32, i32) -> i32
    %alloc_eps = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x1xf32>
    %12 = "builtin.unrealized_conversion_cast"(%alloc_eps) : (memref<1x128x1xf32>) -> !emitc.ptr<f32>
    "linalg.generic"(%alloc_mean, %cst_1, %alloc_eps) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb2(%in : f32, %in_1 : f64, %out : f32):
      %13 = "arith.truncf"(%in_1) : (f64) -> f32
      %14 = "emitc.add"(%in, %13) : (f32, f32) -> f32
      "linalg.yield"(%14) : (f32) -> ()
    }) : (memref<1x128x1xf32>, memref<f64>, memref<1x128x1xf32>) -> ()
    %alloc_rsqrt = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x1xf32>
    %15 = "builtin.unrealized_conversion_cast"(%alloc_rsqrt) : (memref<1x128x1xf32>) -> !emitc.ptr<f32>
    %16 = "emitc.call_opaque"(%alloc_eps, %alloc_rsqrt) <{callee = "lut_squareroot", args = ["Tensor*", "Tensor*"]}> : (memref<1x128x1xf32>, memref<1x128x1xf32>) -> memref<1x128x1xf32>
    %alloc_normalized = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %17 = "builtin.unrealized_conversion_cast"(%alloc_normalized) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %18 = "emitc_ext.constant"() {value = 40 : i32} : () -> i32
    %19 = "emitc.call_opaque"(%arg0, %alloc_rsqrt, %alloc_normalized, %18) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (!emitc.ptr<f32>, memref<1x128x1xf32>, memref<1x128x2048xf32>, i32) -> i32
    %alloc_output = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x128x2048xf32>
    %20 = "builtin.unrealized_conversion_cast"(%alloc_output) : (memref<1x128x2048xf32>) -> !emitc.ptr<f32>
    %21 = "emitc_ext.constant"() {value = 40 : i32} : () -> i32
    %22 = "emitc.call_opaque"(%gamma, %alloc_normalized, %alloc_output, %21) <{callee = "tensor_tensor_operator", args = ["Tensor*", "Tensor*", "Tensor*", "uint32_t"]}> : (memref<2048xf32>, memref<1x128x2048xf32>, memref<1x128x2048xf32>, i32) -> i32
    "func.return"(%alloc_output) : (memref<1x128x2048xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "RMSNormTest"} : () -> ()
