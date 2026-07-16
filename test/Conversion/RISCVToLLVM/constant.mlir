// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
        %0 = "rair.constant"() {value = 0 : index} : () -> index
        %1 = "rair.constant"() {value = 42 : i32} : () -> i32
        %6 = "rair.constant"() {value = 4 : i4} : () -> i4
        %7 = "rair.constant"() {value = 42 : i8} : () -> i8
        %8 = "rair.constant"() {value = 42 : i16} : () -> i16


        %2 = "rair.constant"() {value = 42 : i64} : () -> i64
        %3 = "rair.constant"() {value = 0.000000e+00 : f32} : () -> f32
        %4 = "rair.constant"() {value = 1.000000e+00 : f16} : () -> f16
        %9 = "rair.constant"()  { value = dense<5.5> : tensor<f64> } : () -> tensor<f64>


    return
}