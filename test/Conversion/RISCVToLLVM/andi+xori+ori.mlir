// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
//Scalar Integer Constants
  %c0_i32 = "rocc.constant"() {value = 0 : i32} : () -> i32
  %c1_i32 = "rocc.constant"() {value = 1 : i32} : () -> i32

  //  tensor Integer Constants
  %vec2x3_i16_0 = "rocc.constant"() {
    value = dense<[[1, 0, 0], [0, 0, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>
 
  %vec2x3_i16_1 = "rocc.constant"() {
    value = dense<[[1, 0, 1], [0, 1, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>

  //  Bitwise AND Operations (rocc.andi)
  %res_scalar_i32_andi = "rocc.andi"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_andi = "rocc.andi"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rocc.print"(%res_scalar_i32_andi) : (i32) -> ()
  // "rocc.print"(%res_vec2x3_i16_andi) : (tensor<2x3xi16>) -> ()



  // Bitwise XOR Operations (rocc.xori)
  %res_scalar_i32_xori = "rocc.xori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_xori = "rocc.xori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rocc.print"(%res_scalar_i32_xori) : (i32) -> ()
  // "rocc.print"(%res_vec2x3_i16_xori) : (tensor<2x3xi16>) -> ()

  //  Bitwise OR Operations (rocc.ori) 
  %res_scalar_i32_ori = "rocc.ori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_ori = "rocc.ori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "rocc.print"(%res_scalar_i32_ori) : (i32) -> ()
  // "rocc.print"(%res_vec2x3_i16_ori) : (tensor<2x3xi16>) -> ()
  return
}