// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s
// CHECK: llvm.func @main()
func.func @main() {
        %0 = "riscv.constant"() {value = dense<
        [
            [[1.1, 2.2, 3.3, 4.4],   
             [5.5, 6.6, 7.7, 8.8]],
            
            [[9.9, 10.1, 11.2, 12.3], 
             [13.4, 14.5, 15.6, 16.7]],
            
            [[17.8, 18.9, 19.0, 20.1], 
             [21.2, 22.3, 23.4, 24.5]]
        ]
    > : tensor<3x2x4xf64>} : () -> tensor<3x2x4xf64>
    %1 = "riscv.constant"() {value = dense<
        [[1.1, 2.2, 3.3, 4.4],   
             [5.5, 6.6, 7.7, 8.8]
        ]
    > : tensor<2x4xf64>} : () -> tensor<2x4xf64>
    %res1 = "riscv.reduce"(%0) {kind = "sum",dim=[0]} : (tensor<3x2x4xf64>) -> tensor<2x4xf64>

    "riscv.print"(%res1) : (tensor<2x4xf64>) -> ()

    return
}
