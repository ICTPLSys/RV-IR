// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()

func.func @main() {
    %0 = "rocc.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    %shape = "rocc.constant"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    // %1 = "rocc.reshape"(%0, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %1 = "rocc.reshape"(%0, %shape) : (tensor<2x3xf64>, tensor<2xi32>) -> tensor<3x2xf64>

    "rocc.print"(%1) : (tensor<3x2xf64>) -> ()
    return
}
