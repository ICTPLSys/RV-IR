add_mlir_library(MLIRRAIRTransforms
  Passes.cpp
  VerifyLifetimes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/RISCV/Transforms

  DEPENDS
  RAIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRRISCVDialect
  MLIRSupport
)

torch_mlir_target_includes(MLIRRAIRTransforms)
