// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
//Scalar Integer Constants
  %c0_i32 = "rair.constant"() {value = 0 : i32} : () -> i32
  %c1_i32 = "rair.constant"() {value = 1 : i32} : () -> i32

  //  tensor Integer Constants
  %vec2x3_i16_0 = "rair.constant"() {
    value = dense<[[1, 0, 0], [0, 0, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>
 
  %vec2x3_i16_1 = "rair.constant"() {
    value = dense<[[1, 0, 1], [0, 1, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>

  //  Bitwise AND Operations (rair.andi)
  %res_scalar_i32_andi = "rair.andi"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_andi = "rair.andi"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rair.print"(%res_scalar_i32_andi) : (i32) -> ()
  // "rair.print"(%res_vec2x3_i16_andi) : (tensor<2x3xi16>) -> ()



  // Bitwise XOR Operations (rair.xori)
  %res_scalar_i32_xori = "rair.xori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_xori = "rair.xori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rair.print"(%res_scalar_i32_xori) : (i32) -> ()
  // "rair.print"(%res_vec2x3_i16_xori) : (tensor<2x3xi16>) -> ()

  //  Bitwise OR Operations (rair.ori) 
  %res_scalar_i32_ori = "rair.ori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_ori = "rair.ori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rair.print"(%res_scalar_i32_ori) : (i32) -> ()
  // "rair.print"(%res_vec2x3_i16_ori) : (tensor<2x3xi16>) -> ()
  return
}