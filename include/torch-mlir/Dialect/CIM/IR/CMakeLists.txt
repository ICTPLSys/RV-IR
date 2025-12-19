#add_mlir_dialect(CIMOps cim)
#add_mlir_doc(CIMDialect CIMDialect CIM/ -gen-dialect-doc)
#add_mlir_doc(CIMOps CIMOps CIM/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CIMOps.td)
mlir_tablegen(CIMOps.h.inc -gen-op-decls)
mlir_tablegen(CIMOps.cpp.inc -gen-op-defs)
mlir_tablegen(CIMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CIMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRCIMOpsIncGen)
add_dependencies(mlir-headers MLIRCIMOpsIncGen)

add_mlir_doc(CIMDialect CIMDialect CIM/ -gen-dialect-doc)
add_mlir_doc(CIMOps CIMOps CIM/ -gen-op-doc)