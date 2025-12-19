#ifndef RISCV_CONVERSION_PASSES
#define RISCV_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RISCV conversions
//===----------------------------------------------------------------------===//


def ConvertRISCVToCIMPass : Pass<"convert-riscv-to-cim", operation="func::FuncOp"> {
    let summary = "Convert linalg to riscv dialect";
    let constructor = "cim::createRISCVLowerToCIMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","linalg::LinalgDialect"];
}

def ConvertCIMToLLVMPass : Pass<"convert-cim-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert cim to llvm dialect";
    let constructor = "cim::createCIMLowerToLLVMPass()";
    let dependentDialects = [ "cim::CIMDialect", "tensor::TensorDialect","llvm::LLVMDialect", "memref::MemRefDialect"];
}

def ConvertLinalgToRISCVPass : Pass<"convert-linalg-to-riscv", operation="func::FuncOp"> {
    let summary = "Convert linalg to riscv dialect";
    let constructor = "riscv::createLowerLinalgToRISCVPass()";
    let dependentDialects = [ "riscv::RISCVDialect", "tensor::TensorDialect"];
}

def ConvertRiscvToAffinePass : Pass<"convert-riscv-to-affine", operation="func::FuncOp"> {
    let summary = "Convert RISC-V dialect to Affine dialect";
    let constructor = "riscv::createLowerToAffinePass()";
    let dependentDialects = [
        "riscv::RISCVDialect",
        "affine::AffineDialect",  
        "bufferization::BufferizationDialect", 
        "memref::MemRefDialect", 
        "tensor::TensorDialect"
    ];
}


def ConvertRiscvToLLVMPass : Pass<"convert-riscv-to-llvm", operation="func::FuncOp"> {
    let summary = "Convert RISC-V dialect to LLVM dialect";
    let constructor = "riscv::createLowerToLLVMPass()";
    let dependentDialects = [
        "riscv::RISCVDialect",
        "llvm::LLVMDialect",  
        "func::FuncDialect",
        "memref::MemRefDialect"
    ];
}

#endif // RISCV_CONVERSION_PASSES
