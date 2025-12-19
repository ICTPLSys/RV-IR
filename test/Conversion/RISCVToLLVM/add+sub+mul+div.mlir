// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
  // Integer constants of different bit widths
  %c0_idx  = "riscv.constant"() {value = 0 : index} : () -> index
  %c42_i32 = "riscv.constant"() {value = 42 : i32}  : () -> i32
  %c11_i32 = "riscv.constant"() {value = 11 : i32}  : () -> i32
  
  %c4_i4   = "riscv.constant"() {value = 4 : i4}    : () -> i4
  %c8_i4   = "riscv.constant"() {value = 8 : i4}    : () -> i4
  
  %c42_i8  = "riscv.constant"() {value = 42 : i8}   : () -> i8
  %c42_i16 = "riscv.constant"() {value = 42 : i16}  : () -> i16
  %c22_i16 = "riscv.constant"() {value = 22 : i16}  : () -> i16
  
  %c42_i64 = "riscv.constant"() {value = 42 : i64}  : () -> i64

  // Floating-point constants of different formats
  %c0p0_f32  = "riscv.constant"() { value = dense<5.5> : tensor<f32>}  : () -> tensor<f32>
  %c1p0_f32  = "riscv.constant"() { value = dense<5.5> : tensor<f32>}  : () -> tensor<f32>
  
  %c1p0_f16  = "riscv.constant"() {value = 1.000000e+00 : f16}  : () -> f16
  %c4p0_f16  = "riscv.constant"() {value = 4.000000e+00 : f16}  : () -> f16
  %c4p0_bf16 = "riscv.constant"() {value = 4.000000e+00 : bf16} : () -> bf16

  // tensor constants (floating-point)
  %vec2x3_f16  = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>}  : () -> tensor<2x3xf16>
  %vec2x3_f16_2 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16>} : () -> tensor<2x3xf16>
  %vec2x3_bf16 = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xbf16>} : () -> tensor<2x3xbf16>
  %vec2x3_f32  = "riscv.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>}  : () -> tensor<2x3xf32>

  // tensor constants (integer)
  %vec2x3_i16 = "riscv.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>} : () -> tensor<2x3xi16>
  %vec2x3_i16_2 = "riscv.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>} : () -> tensor<2x3xi16>
  %vec1x3_i32 = "riscv.constant"() {value = dense<[[1, 2, 3]]> : tensor<1x3xi32>} : () -> tensor<1x3xi32>


  // Test cases: arithmetic operations
  // 1. Scalar floating-point addition
  %res_f32_add = "riscv.add"(%c0p0_f32, %c1p0_f32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // "riscv.print"(%res_f32_add) : (tensor<f32>) -> ()

  
  // 2. tensor floating-point addition (same type)
  %res_vec_f16_add = "riscv.add"(%vec2x3_f16, %vec2x3_f16_2) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  // "riscv.print"(%res_vec_f16_add) : (tensor<2x3xf16>) -> ()

  
  // 3. Scalar integer addition
  %res_i32_add = "riscv.add"(%c42_i32, %c11_i32) : (i32, i32) -> i32
  
  // 4. tensor integer addition (same type)
  %res_vec_i16_add = "riscv.add"(%vec2x3_i16, %vec2x3_i16_2) : (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  
  // 5. Other arithmetic operations
  %res_f16_sub = "riscv.sub"(%c1p0_f16, %c4p0_f16) : (f16, f16) -> f16
  %res_vec_f16_sub = "riscv.sub"(%vec2x3_f16, %vec2x3_f16_2) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>

  %res_f16_mul = "riscv.mul"(%c1p0_f16, %c4p0_f16) : (f16, f16) -> f16
  %res_vec_f16_mul = "riscv.mul"(%vec2x3_f16, %vec2x3_f16_2) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>

  %res_f16_div = "riscv.div"(%c1p0_f16, %c4p0_f16) : (f16, f16) -> f16
  %res_vec_f16_div = "riscv.div"(%vec2x3_f16, %vec2x3_f16_2) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>


  return
}