// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()

func.func @main() {
    // Create integer constants for comparison
    %const_i32_0 = "riscv.constant"() {value = 0 : i32} : () -> i32  // Constant value 0 of type i32
    %const_i32_1 = "riscv.constant"() {value = 1 : i32} : () -> i32  // Constant value 1 of type i32

    // Create 2D integer tensor constants 
    %const_tensor_2x3i16_a = "riscv.constant"()  { 
        value = dense<[[1, 0, 0], [0, 0, 0]]> : tensor<2x3xi16> 
    } : () -> tensor<2x3xi16>  
    
    %const_tensor_2x3i16_b = "riscv.constant"()  { 
        value = dense<[[1, 2, 1], [0, 1, 0]]> : tensor<2x3xi16> 
    } : () -> tensor<2x3xi16> 

    // Compute maximum value between two i32 constants
    %max_i32_0_1 = "riscv.max"(%const_i32_0, %const_i32_1) : (i32, i32) -> i32  
    // Compute minimum value between two i32 constants
    %min_i32_0_1 = "riscv.min"(%const_i32_0, %const_i32_1) : (i32, i32) -> i32  
    // "riscv.print"(%max_i32_0_1) : (i32) -> ()
    // "riscv.print"(%min_i32_0_1) : (i32) -> ()

        
    return 
}