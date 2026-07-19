add_mlir_library(MLIRRAIRDialect
  RAIRDialect.cpp
  RAIROps.cpp  

  #${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/RAIR/IR/RAIROps.cpp.inc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/RAIR
  ${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/RAIR/IR

  DEPENDS
  MLIRRAIROpsIncGen
  RAIROpInterfacesIncGen
  RAIRTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRInferTypeOpInterface
)
torch_mlir_target_includes(MLIRRAIRDialect)
