set(LLVM_TARGET_DEFINITIONS RISCVPasses.td)
mlir_tablegen(RISCVPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(RAIRTransformsPassIncGen)

add_mlir_doc(RISCVPasses RAIRTransforms ./ -gen-pass-doc)
