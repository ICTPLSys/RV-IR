add_subdirectory(TorchOnnxToTorch)

set(LLVM_TARGET_DEFINITIONS RISCVPasses.td)
mlir_tablegen(RISCVPasses.h.inc -gen-pass-decls -name RISCVConversion ${TORCH_MLIR_TABLEGEN_FLAGS})
add_public_tablegen_target(RISCVConversionPassIncGen)

add_mlir_doc(RISCVPasses RISCVConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls ${TORCH_MLIR_TABLEGEN_FLAGS})

add_public_tablegen_target(TorchMLIRConversionPassIncGen)

add_mlir_doc(Passes TorchMLIRConversionPasses ./ -gen-pass-doc)



