set(LLVM_TARGET_DEFINITIONS RAIRPasses.td)
mlir_tablegen(RAIRPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(RAIRTransformsPassIncGen)

add_mlir_doc(RAIRPasses RAIRTransforms ./ -gen-pass-doc)
