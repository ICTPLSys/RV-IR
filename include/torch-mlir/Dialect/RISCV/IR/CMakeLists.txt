set(LLVM_TARGET_DEFINITIONS RISCVDialect.td)
mlir_tablegen(RISCVOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RISCVOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(RISCVEnums.h.inc -gen-enum-decls)
mlir_tablegen(RISCVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(RISCVOpInterfacesIncGen)

#add_mlir_dialect(RISCVOps riscv)
#add_mlir_doc(RISCVDialect RISCVDialect RISCV/ -gen-dialect-doc)
#add_mlir_doc(RISCVOps RISCVOps RISCV/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RISCVOps.td)
mlir_tablegen(RISCVOps.h.inc -gen-op-decls)
mlir_tablegen(RISCVOps.cpp.inc -gen-op-defs)
mlir_tablegen(RISCVDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RISCVDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRRISCVOpsIncGen)
add_dependencies(mlir-headers MLIRRISCVOpsIncGen)

add_mlir_doc(RISCVDialect RISCVDialect RISCV/ -gen-dialect-doc)
add_mlir_doc(RISCVOps RISCVOps RISCV/ -gen-op-doc)

