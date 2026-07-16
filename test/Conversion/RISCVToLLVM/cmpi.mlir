// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {

    %t1 = "rair.constant"() {value = 42 : i32} : () -> i32
    %t2 = "rair.constant"() {value = 32 : i32} : () -> i32
    %5 = "rair.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %6 = "rair.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %7 = "rair.constant"()  { value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>
    %77 = "rair.constant"()  { value = dense<[[1, 2, 3], [4, 5, 9]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>


    // elementwise eq
    %t3 = "rair.cmpi"(%t1, %t2) {predicate = "eq"} 
          : (i32, i32) -> i1
    %t4 = "rair.cmpi"(%5, %6) {predicate = "eq"} 
          : (tensor<i16>, tensor<i16>) ->  tensor<i1>
          
    %t5 = "rair.cmpi"(%7, %77) {predicate = "ne"} 
        : (tensor<2x3xi16>, tensor<2x3xi16>) ->  tensor<2x3xi1>
    // "rair.print"(%t3) : (i1) -> ()
    // "rair.print"(%t4) : (tensor<i1>) -> ()
    // "rair.print"(%t5) : (tensor<2x3xi1>) -> ()

    
    return 
}