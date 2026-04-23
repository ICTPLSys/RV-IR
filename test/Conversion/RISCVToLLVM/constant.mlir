// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
        %0 = "rocc.constant"() {value = 0 : index} : () -> index
        %1 = "rocc.constant"() {value = 42 : i32} : () -> i32
        %6 = "rocc.constant"() {value = 4 : i4} : () -> i4
        %7 = "rocc.constant"() {value = 42 : i8} : () -> i8
        %8 = "rocc.constant"() {value = 42 : i16} : () -> i16


        %2 = "rocc.constant"() {value = 42 : i64} : () -> i64
        %3 = "rocc.constant"() {value = 0.000000e+00 : f32} : () -> f32
        %4 = "rocc.constant"() {value = 1.000000e+00 : f16} : () -> f16
        %9 = "rocc.constant"()  { value = dense<5.5> : tensor<f64> } : () -> tensor<f64>


    return
}