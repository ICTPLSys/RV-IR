// RUN: torch-mlir-opt <%s -convert-linalg-to-riscv -convert-riscv-to-affine -convert-riscv-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
    // Create input matrices
    %input = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],[13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf64>
    %kernel = arith.constant dense<[[0.1, 0.2],[0.3, 0.4]]> : tensor<2x2xf64>
    
    %output = arith.constant dense<0.0> : tensor<3x3xf64>
    
    // Perform matrix multiplication using linalg.matmul
    %res = linalg.conv_2d ins(%input, %kernel: tensor<4x4xf64>, tensor<2x2xf64>) 
                       outs(%output : tensor<3x3xf64>) -> tensor<3x3xf64>
    
    // Print results
    "riscv.print"(%res) : (tensor<3x3xf64>) -> ()
    
    return
}