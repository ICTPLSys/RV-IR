#ifndef RISCV_CONVERSION_PASSES
#define RISCV_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RISCV conversions
//===----------------------------------------------------------------------===//


def ConvertRAIRToCIMPass : Pass<"convert-rair-to-cim", operation="func::FuncOp"> {
    let summary = "Convert RAIR dialect to CIM dialect";
    let constructor = "cim::createRAIRLowerToCIMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","linalg::LinalgDialect"];
}

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

def ConvertRAIRToAffinePass : Pass<"convert-rair-to-affine", operation="func::FuncOp"> {
    let summary = "Convert RAIR dialect to Affine dialect";
    let constructor = "rair::createLowerToAffinePass()";
    let dependentDialects = [
        "rair::RAIRDialect",
        "affine::AffineDialect",  
        "bufferization::BufferizationDialect", 
        "memref::MemRefDialect", 
        "tensor::TensorDialect"
    ];
}


def ConvertRAIRToLLVMPass : Pass<"convert-rair-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert RAIR dialect to LLVM dialect";
    let constructor = "rair::createLowerToLLVMPass()";
    let dependentDialects = [
        "rair::RAIRDialect",
        "llvm::LLVMDialect",  
        "func::FuncDialect",
        "memref::MemRefDialect"
    ];
}

#endif // RISCV_CONVERSION_PASSES
