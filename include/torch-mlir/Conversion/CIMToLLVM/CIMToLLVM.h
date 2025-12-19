

#include <torch-mlir/Dialect/CIM/IR/CIMDialect.h>
#include <mlir/Pass/Pass.h>

namespace cim {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createCIMLowerToLLVMPass();
} // namespace riscv