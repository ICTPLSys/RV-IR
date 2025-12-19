add_mlir_conversion_library(MLIRRISCVToCIM
    RISCVToCIM.cpp


    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/RISCVToCIM

    DEPENDS
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
    MLIRCIMDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
)
torch_mlir_target_includes(MLIRRISCVToCIM)
