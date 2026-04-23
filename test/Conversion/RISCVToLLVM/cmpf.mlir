// RUN: torch-mlir-opt <%s --convert-rocc-to-affine --convert-rocc-to-llvm | FileCheck %s

// CHECK: llvm.func @main()
func.func @main() {
    // %t1 = "rocc.constant"() {
    //     value = dense<[ [1, 2, 3], [4, 5, 6] ]> : tensor<2x3xi32>
    // } : () -> tensor<2x3xi32>

    // %t2 = "rocc.constant"() {
    //     value = dense<[ [1, 2, 0], [4, 0, 6] ]> : tensor<2x3xi32>
    // } : () -> tensor<2x3xi32>
    %t1 = "rocc.constant"() {value = 42.0 : f32} : () -> f32
    %t2 = "rocc.constant"() {value = 32.0 : f32} : () -> f32
    %5 = "rocc.constant"()  { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16> } : () -> tensor<2x3xf16>
    %55 = "rocc.constant"()  { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf16> } : () -> tensor<2x3xf16>


    // elementwise eq
    %t3 = "rocc.cmpf"(%t1, %t2) {predicate = "eq"} 
          : (f32, f32) -> i1
    %t4 = "rocc.cmpf"(%5, %55) {predicate = "eq"} 
          : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xi1>
          // "rocc.print"(%t3) : (i1) -> ()
    // "rocc.print"(%t3) : (tensor<i1>) -> ()
    // "rocc.print"(%t4) : (tensor<2x3xi1>) -> ()

        
    return 
}