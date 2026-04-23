// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {

    %t1 = "rocc.constant"() {value = 42 : i32} : () -> i32
    %t2 = "rocc.constant"() {value = 32 : i32} : () -> i32
    %5 = "rocc.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %6 = "rocc.constant"()  { value = dense<5> : tensor<i16> } : () -> tensor<i16>
    %7 = "rocc.constant"()  { value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>
    %77 = "rocc.constant"()  { value = dense<[[1, 2, 3], [4, 5, 9]]> : tensor<2x3xi16> } : () -> tensor<2x3xi16>


    // elementwise eq
    %t3 = "rocc.cmpi"(%t1, %t2) {predicate = "eq"} 
          : (i32, i32) -> i1
    %t4 = "rocc.cmpi"(%5, %6) {predicate = "eq"} 
          : (tensor<i16>, tensor<i16>) ->  tensor<i1>
          
    %t5 = "rocc.cmpi"(%7, %77) {predicate = "ne"} 
        : (tensor<2x3xi16>, tensor<2x3xi16>) ->  tensor<2x3xi1>
    // "rocc.print"(%t3) : (i1) -> ()
    // "rocc.print"(%t4) : (tensor<i1>) -> ()
    // "rocc.print"(%t5) : (tensor<2x3xi1>) -> ()

    
    return 
}