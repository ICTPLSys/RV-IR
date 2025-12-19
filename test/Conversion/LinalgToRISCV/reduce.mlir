// RUN: torch-mlir-opt <%s -convert-linalg-to-riscv -convert-riscv-to-affine -convert-riscv-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
    // Create input tensor
    %input = arith.constant dense<[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]> : tensor<2x3xf64>

    // Print input
    "riscv.print"(%input) : (tensor<2x3xf64>) -> ()

    // Initialize reduction output (for sum along dimension 0 → tensor<3xf64>)
    %init = arith.constant dense<0.0> : tensor<3xf64>

    // Perform reduction (sum along dimension 0)
    %sum = linalg.reduce
        ins(%input : tensor<2x3xf64>)
        outs(%init : tensor<3xf64>)
        dimensions =[0]
            (%in: f64, %out: f64){
                %r = arith.addf %out, %in : f64
                linalg.yield %r : f64
        } 

    // Print reduced result
    "riscv.print"(%sum) : (tensor<3xf64>) -> ()

    return
}
