add_mlir_conversion_library(MLIRRISCVToAffine
    LowerToAffine.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/RISCVToAffine

    DEPENDS
    #RISCVIncGen
    MLIRRISCVOpsIncGen
    #RISCVConversionPassIncGen

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

torch_mlir_target_includes(MLIRRISCVToAffine)
