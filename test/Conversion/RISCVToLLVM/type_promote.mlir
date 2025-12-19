// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    // Integer scalar constants (various bit widths)

    %c0_idx  = "riscv.constant"() {value = 0 : index} : () -> index
    %c42_i32 = "riscv.constant"() {value = 42 : i32} : () -> i32
    %c11_i32 = "riscv.constant"() {value = 11 : i32} : () -> i32
    %c4_i4   = "riscv.constant"() {value = 4 : i4} : () -> i4
    %c8_i4   = "riscv.constant"() {value = 8 : i4} : () -> i4
    %c42_i8  = "riscv.constant"() {value = 42 : i8} : () -> i8
    %c42_i16 = "riscv.constant"() {value = 42 : i16} : () -> i16
    %c22_i16 = "riscv.constant"() {value = 22 : i16} : () -> i16
    %c42_i64 = "riscv.constant"() {value = 42 : i64} : () -> i64

    // Floating-point scalar constants (various formats)
    %c0p0_f32  = "riscv.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %c1p0_f32  = "riscv.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %c1p0_f16  = "riscv.constant"() {value = 1.000000e+00 : f16} : () -> f16
    %c4p0_f16  = "riscv.constant"() {value = 4.000000e+00 : f16} : () -> f16
    %c4p0_bf16 = "riscv.constant"() {value = 4.000000e+00 : bf16} : () -> bf16

    // Floating-point tensor constants (various formats and shapes)
    %vec2x3_f16  = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>} : () -> tensor<2x3xf16>
    %vec2x3_f16_2 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>} : () -> tensor<2x3xf16>
    %vec2x3_bf16 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xbf16>} : () -> tensor<2x3xbf16>
    %vec2x4_bf16 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]> : tensor<2x4xbf16>} : () -> tensor<2x4xbf16>
    %vec2x3_f32  = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>

    // Integer tensor constants (various bit widths)
    %vec2x3_i16 = "riscv.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>} : () -> tensor<2x3xi16>
    %vec2x3_i16_2 = "riscv.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>} : () -> tensor<2x3xi16>
    %vec2x3_i32 = "riscv.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    %vec2x2_i32 = "riscv.constant"() {value = dense<[[1, 2], [4, 5]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    // Type promotion test cases

    // 1. Scalar type promotion
    %res_scalar_bf16_i32 = "riscv.add"(%c4p0_bf16, %c42_i32) : (bf16, i32) -> bf16  // float > int
    %res_scalar_i32_bf16 = "riscv.add"(%c42_i32, %c4p0_bf16) : (i32, bf16) -> bf16  // float > int
    %res_scalar_f32_f16  = "riscv.add"(%c1p0_f32, %c1p0_f16) : (f32, f16) -> f32    // wider float
    %res_scalar_f16_f32  = "riscv.add"(%c1p0_f16, %c1p0_f32) : (f16, f32) -> f32    // wider float
    %res_scalar_f16_bf16 = "riscv.add"(%c4p0_f16, %c4p0_bf16) : (f16, bf16) -> f16  // prefer f16
    %res_scalar_bf16_f16 = "riscv.add"(%c4p0_bf16, %c4p0_f16) : (bf16, f16) -> f16  // prefer f16
    // "riscv.print"(%res_scalar_bf16_i32) : (bf16) -> ()
    // "riscv.print"(%res_scalar_i32_bf16) : (bf16) -> ()  
    // "riscv.print"(%res_scalar_f32_f16)  : (f32) -> ()  
    // "riscv.print"(%res_scalar_f16_f32)  : (f32) -> ()  
    // "riscv.print"(%res_scalar_f16_bf16) : (f16) -> ()  
    // "riscv.print"(%res_scalar_bf16_f16) : (f16) -> ()  

    

    // 2. tensor type promotion
    %res_vec_bf16_i32 = "riscv.add"(%vec2x3_bf16, %vec2x3_i32) : (tensor<2x3xbf16>, tensor<2x3xi32>) -> tensor<2x3xbf16>  // float > int
    %res_vec_i32_bf16 = "riscv.add"(%vec2x3_i32, %vec2x3_bf16) : (tensor<2x3xi32>, tensor<2x3xbf16>) -> tensor<2x3xbf16>  // float > int
    %res_vec_f32_f16  = "riscv.add"(%vec2x3_f32, %vec2x3_f16) : (tensor<2x3xf32>, tensor<2x3xf16>) -> tensor<2x3xf32>    // wider float
    %res_vec_f16_f32  = "riscv.add"(%vec2x3_f16, %vec2x3_f32) : (tensor<2x3xf16>, tensor<2x3xf32>) -> tensor<2x3xf32>    // wider float
    %res_vec_f16_bf16 = "riscv.add"(%vec2x3_f16, %vec2x3_bf16) : (tensor<2x3xf16>, tensor<2x3xbf16>) -> tensor<2x3xf16>  // prefer f16
    %res_vec_bf16_f16 = "riscv.add"(%vec2x3_bf16, %vec2x3_f16) : (tensor<2x3xbf16>, tensor<2x3xf16>) -> tensor<2x3xf16>  // prefer f16

    // "riscv.print"(%res_vec_bf16_i32) : (tensor<2x3xbf16>) -> ()  
    // "riscv.print"(%res_vec_i32_bf16) : (tensor<2x3xbf16>) -> ()  
    // "riscv.print"(%res_vec_f32_f16)  : (tensor<2x3xf32>) -> () 
    // "riscv.print"(%res_vec_f16_f32)  : (tensor<2x3xf32>) -> ()  
    // "riscv.print"(%res_vec_f16_bf16) : (tensor<2x3xf16>) -> ()  
    // "riscv.print"(%res_vec_bf16_f16) : (tensor<2x3xf16>) -> ()  

    return
}
