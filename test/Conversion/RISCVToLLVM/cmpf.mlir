// RUN: torch-mlir-opt <%s --convert-rair-to-affine --convert-rair-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    // %t1 = "rair.constant"() {
    //     value = dense<[ [1, 2, 3], [4, 5, 6] ]> : tensor<2x3xi32>
    // } : () -> tensor<2x3xi32>

    // %t2 = "rair.constant"() {
    //     value = dense<[ [1, 2, 0], [4, 0, 6] ]> : tensor<2x3xi32>
    // } : () -> tensor<2x3xi32>
    %t1 = "rair.constant"() {value = 42.0 : f32} : () -> f32
    %t2 = "rair.constant"() {value = 32.0 : f32} : () -> f32
    %5 = "rair.constant"()  { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16> } : () -> tensor<2x3xf16>
    %55 = "rair.constant"()  { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16> } : () -> tensor<2x3xf16>


    // elementwise eq
    %t3 = "rair.cmpf"(%t1, %t2) {predicate = "eq"} 
          : (f32, f32) -> i1
    %t4 = "rair.cmpf"(%5, %55) {predicate = "eq"} 
          : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xi1>
          // "rair.print"(%t3) : (i1) -> ()
    // "rair.print"(%t3) : (tensor<i1>) -> ()
    // "rair.print"(%t4) : (tensor<2x3xi1>) -> ()

        
    return 
}