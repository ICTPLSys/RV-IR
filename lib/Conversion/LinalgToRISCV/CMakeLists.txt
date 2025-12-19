add_mlir_conversion_library(MLIRLinalgToRISCV
    LinalgToRISCV.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/LinalgToRISCV

    DEPENDS
    RISCVIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRRISCVDialect   #必须添加，否则出现链接错误
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM

)
torch_mlir_target_includes(MLIRLinalgToRISCV)
