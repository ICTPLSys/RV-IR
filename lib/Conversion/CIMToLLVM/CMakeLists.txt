add_mlir_conversion_library(MLIRCIMToLLVM
    CIMToLLVM.cpp


    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/CIMToLLVM

    DEPENDS
    MLIRCIMOpsIncGen

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
    #MLIRRISCVDialect
    MLIRCIMDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
)
torch_mlir_target_includes(MLIRCIMToLLVM)
