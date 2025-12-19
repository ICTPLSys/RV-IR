add_mlir_conversion_library(MLIRRISCVToLLVM
    LowerToLLVM.cpp


    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/RISCVToLLVM

    DEPENDS
    #RISCVIncGen
    MLIRRISCVOpsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRRISCVDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
)
torch_mlir_target_includes(MLIRRISCVToLLVM)
