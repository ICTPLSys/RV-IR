// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
        %0 = "riscv.constant"() {value = 0 : index} : () -> index
        %1 = "riscv.constant"() {value = 42 : i32} : () -> i32
        %6 = "riscv.constant"() {value = 4 : i4} : () -> i4
        %7 = "riscv.constant"() {value = 42 : i8} : () -> i8
        %8 = "riscv.constant"() {value = 42 : i16} : () -> i16


        %2 = "riscv.constant"() {value = 42 : i64} : () -> i64
        %3 = "riscv.constant"() {value = 0.000000e+00 : f32} : () -> f32
        %4 = "riscv.constant"() {value = 1.000000e+00 : f16} : () -> f16
        %9 = "riscv.constant"()  { value = dense<5.5> : tensor<f64> } : () -> tensor<f64>


    return
}