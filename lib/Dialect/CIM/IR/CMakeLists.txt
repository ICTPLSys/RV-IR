add_mlir_library(MLIRCIMDialect
  CIMDialect.cpp
  CIMOps.cpp  

  #${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/CIM/IR/CIMOps.cpp.inc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/CIM
  ${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/CIM/IR

  DEPENDS
  MLIRCIMOpsIncGen


  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  MLIRInferTypeOpInterface
)
torch_mlir_target_includes(MLIRCIMDialect)
