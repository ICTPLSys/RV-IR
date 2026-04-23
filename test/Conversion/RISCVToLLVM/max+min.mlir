// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()

func.func @main() {
    // Create integer constants for comparison
    %const_i32_0 = "rocc.constant"() {value = 0 : i32} : () -> i32  // Constant value 0 of type i32
    %const_i32_1 = "rocc.constant"() {value = 1 : i32} : () -> i32  // Constant value 1 of type i32

    // Create 2D integer tensor constants 
    %const_tensor_2x3i16_a = "rocc.constant"()  { 
        value = dense<[[1, 0, 0], [0, 0, 0]]> : tensor<2x3xi16> 
    } : () -> tensor<2x3xi16>  
    
    %const_tensor_2x3i16_b = "rocc.constant"()  { 
        value = dense<[[1, 2, 1], [0, 1, 0]]> : tensor<2x3xi16> 
    } : () -> tensor<2x3xi16> 

    // Compute maximum value between two i32 constants
    %max_i32_0_1 = "rocc.max"(%const_i32_0, %const_i32_1) : (i32, i32) -> i32  
    // Compute minimum value between two i32 constants
    %min_i32_0_1 = "rocc.min"(%const_i32_0, %const_i32_1) : (i32, i32) -> i32  
    // "rocc.print"(%max_i32_0_1) : (i32) -> ()
    // "rocc.print"(%min_i32_0_1) : (i32) -> ()

        
    return 
}