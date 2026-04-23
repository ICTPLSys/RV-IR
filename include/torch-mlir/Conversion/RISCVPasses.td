#ifndef RISCV_CONVERSION_PASSES
#define RISCV_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RISCV conversions
//===----------------------------------------------------------------------===//


def ConvertROCCToCIMPass : Pass<"convert-rocc-to-cim", operation="func::FuncOp"> {
    let summary = "Convert ROCC dialect to CIM dialect";
    let constructor = "cim::createROCCLowerToCIMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","linalg::LinalgDialect"];
}

def ConvertCIMToLLVMPass : Pass<"convert-cim-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert cim to llvm dialect";
    let constructor = "cim::createCIMLowerToLLVMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","llvm::LLVMDialect", "memref::MemRefDialect"];
}

def ConvertLinalgToROCCPass : Pass<"convert-linalg-to-rocc", operation="func::FuncOp"> {
    let summary = "Convert Linalg to ROCC dialect";
    let constructor = "rocc::createLowerLinalgToROCCPass()";
    let dependentDialects = [ "rocc::ROCCDialect", "tensor::TensorDialect"];
}

def ConvertROCCToAffinePass : Pass<"convert-rocc-to-affine", operation="func::FuncOp"> {
    let summary = "Convert ROCC dialect to Affine dialect";
    let constructor = "rocc::createLowerToAffinePass()";
    let dependentDialects = [
        "rocc::ROCCDialect",
        "affine::AffineDialect",  
        "bufferization::BufferizationDialect", 
        "memref::MemRefDialect", 
        "tensor::TensorDialect"
    ];
}


def ConvertROCCToLLVMPass : Pass<"convert-rocc-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert ROCC dialect to LLVM dialect";
    let constructor = "rocc::createLowerToLLVMPass()";
    let dependentDialects = [
        "rocc::ROCCDialect",
        "llvm::LLVMDialect",  
        "func::FuncDialect",
        "memref::MemRefDialect"
    ];
}

#endif // RISCV_CONVERSION_PASSES
