// RUN: torch-mlir-opt <%s -convert-linalg-to-riscv -convert-riscv-to-affine -convert-riscv-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
    // Create input matrices
    %A = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %B = arith.constant dense<[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]> : tensor<2x3xf64>
    
    // Create output matrix initialized to zero
    %C_init = arith.constant dense<0.0> : tensor<2x3xf64>
    
    // Perform matrix multiplication using linalg.matmul
    %C = linalg.mul ins(%A, %B : tensor<2x3xf64>, tensor<2x3xf64>) 
                       outs(%C_init : tensor<2x3xf64>) -> tensor<2x3xf64>
    
    // Print results
    "riscv.print"(%A) : (tensor<2x3xf64>) -> ()
    "riscv.print"(%B) : (tensor<2x3xf64>) -> ()
    "riscv.print"(%C) : (tensor<2x3xf64>) -> ()
    
    return
}