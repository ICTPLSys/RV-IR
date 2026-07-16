"builtin.module"() ({
  "func.func"() <{sym_name = "gemmini_matmul_16x16", function_type = (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>}> ({
  ^bb0(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>):
    %cst = "emitc_ext.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %alloc = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16xf32>
    %0 = "builtin.unrealized_conversion_cast"(%alloc) : (memref<16x16xf32>) -> !emitc.ptr<f32>
    "linalg.fill"(%cst, %alloc) <{operandSegmentSizes = array<i32: 1, 1>}> : (f32, memref<16x16xf32>) -> ()
    %1 = "emitc_ext.constant"() {value = 1 : index} : () -> index
    %2 = "emitc_ext.constant"() {value = 16 : index} : () -> index
    %3 = "emitc_ext.constant"() {value = 16 : index} : () -> index
    %4 = "emitc_ext.constant"() {value = 16 : index} : () -> index
    "rair.matmul"(%arg0, %arg1, %alloc) : (!emitc.ptr<f32>, !emitc.ptr<f32>, memref<16x16xf32>) -> ()
    "func.return"(%alloc) : (memref<16x16xf32>) -> ()
  }) : () -> ()
}) {torch.debug_module_name = "MatmulGemmini16"} : () -> ()