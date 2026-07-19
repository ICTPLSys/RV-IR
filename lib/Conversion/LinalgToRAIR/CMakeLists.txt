add_mlir_conversion_library(MLIRLinalgToRAIR
    LinalgToRAIR.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/LinalgToRAIR

    DEPENDS
    RAIRIncGen

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
    MLIRRAIRDialect   #必须添加，否则出现链接错误
    MLIRLinalgToStandard
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM

)
torch_mlir_target_includes(MLIRLinalgToRAIR)
