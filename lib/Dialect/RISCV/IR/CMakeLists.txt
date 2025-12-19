add_mlir_library(MLIRRISCVDialect
  RISCVDialect.cpp
  RISCVOps.cpp  

  #${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/RISCV/IR/RISCVOps.cpp.inc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/RISCV
  ${PROJECT_BINARY_DIR}/include/torch-mlir/Dialect/RISCV/IR

  DEPENDS
  MLIRRISCVOpsIncGen
  RISCVOpInterfacesIncGen
  #TorchMLIRRISCVOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  MLIRInferTypeOpInterface
)
torch_mlir_target_includes(MLIRRISCVDialect)
