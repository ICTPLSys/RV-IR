add_mlir_library(MLIRRAIRTransforms
  Passes.cpp
  MaterializePlan.cpp
  MaterializeStaticMatmul.cpp
  PrintEffectReport.cpp
  RAIRPlanAnalysis.cpp
  RAIRStaticEffectGraph.cpp
  VerifyPlan.cpp
  VerifyLifetimes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/RAIR/Transforms

  DEPENDS
  RAIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRRAIRDialect
  MLIRSupport
)

torch_mlir_target_includes(MLIRRAIRTransforms)
