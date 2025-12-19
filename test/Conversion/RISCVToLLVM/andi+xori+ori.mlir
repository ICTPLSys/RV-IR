// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
//Scalar Integer Constants
  %c0_i32 = "riscv.constant"() {value = 0 : i32} : () -> i32
  %c1_i32 = "riscv.constant"() {value = 1 : i32} : () -> i32

  //  tensor Integer Constants
  %vec2x3_i16_0 = "riscv.constant"() {
    value = dense<[[1, 0, 0], [0, 0, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>
 
  %vec2x3_i16_1 = "riscv.constant"() {
    value = dense<[[1, 0, 1], [0, 1, 0]]> : tensor<2x3xi16>
  } : () -> tensor<2x3xi16>

  //  Bitwise AND Operations (riscv.andi)
  %res_scalar_i32_andi = "riscv.andi"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_andi = "riscv.andi"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "riscv.print"(%res_scalar_i32_andi) : (i32) -> ()
  // "riscv.print"(%res_vec2x3_i16_andi) : (tensor<2x3xi16>) -> ()



  // Bitwise XOR Operations (riscv.xori)
  %res_scalar_i32_xori = "riscv.xori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_xori = "riscv.xori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "riscv.print"(%res_scalar_i32_xori) : (i32) -> ()
  // "riscv.print"(%res_vec2x3_i16_xori) : (tensor<2x3xi16>) -> ()

  //  Bitwise OR Operations (riscv.ori) 
  %res_scalar_i32_ori = "riscv.ori"(%c0_i32, %c1_i32) : (i32, i32) -> i32
  %res_vec2x3_i16_ori = "riscv.ori"(%vec2x3_i16_0, %vec2x3_i16_1) : 
    (tensor<2x3xi16>, tensor<2x3xi16>) -> tensor<2x3xi16>
  // "riscv.print"(%res_scalar_i32_ori) : (i32) -> ()
  // "riscv.print"(%res_vec2x3_i16_ori) : (tensor<2x3xi16>) -> ()
  return
}