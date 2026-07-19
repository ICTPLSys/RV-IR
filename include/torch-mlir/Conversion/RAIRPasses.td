#ifndef RAIR_CONVERSION_PASSES
#define RAIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RAIR conversions
//===----------------------------------------------------------------------===//


def ConvertCIMToLLVMPass : Pass<"convert-cim-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert cim to llvm dialect";
    let constructor = "cim::createCIMLowerToLLVMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","llvm::LLVMDialect", "memref::MemRefDialect"];
}

def ConvertLinalgToRAIRPass : Pass<"convert-linalg-to-rair", operation="func::FuncOp"> {
    let summary = "Convert Linalg to RAIR dialect";
    let constructor = "rair::createLowerLinalgToRAIRPass()";
    let dependentDialects = [ "rair::RAIRDialect", "tensor::TensorDialect"];
}

#endif // RAIR_CONVERSION_PASSES
