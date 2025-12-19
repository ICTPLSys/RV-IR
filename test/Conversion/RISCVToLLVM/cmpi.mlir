// RUN: torch-mlir-opt <%s --convert-riscv-to-affine --convert-riscv-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {

    %t1 = "riscv.constant"() {value = 42 : i32} : () -> i32
    %t2 = "riscv.constant"() {value = 32 : i32} : () -> i32
    %5 = "riscv.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %6 = "riscv.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %7 = "riscv.constant"()  { value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>
    %77 = "riscv.constant"()  { value = dense<[[1, 2, 3], [4, 5, 9]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>


    // elementwise eq
    %t3 = "riscv.cmpi"(%t1, %t2) {predicate = "eq"} 
          : (i32, i32) -> i1
    %t4 = "riscv.cmpi"(%5, %6) {predicate = "eq"} 
          : (tensor<i16>, tensor<i16>) ->  tensor<i1>
          
    %t5 = "riscv.cmpi"(%7, %77) {predicate = "ne"} 
        : (tensor<2x3xi16>, tensor<2x3xi16>) ->  tensor<2x3xi1>
    // "riscv.print"(%t3) : (i1) -> ()
    // "riscv.print"(%t4) : (tensor<i1>) -> ()
    // "riscv.print"(%t5) : (tensor<2x3xi1>) -> ()

    
    return 
}