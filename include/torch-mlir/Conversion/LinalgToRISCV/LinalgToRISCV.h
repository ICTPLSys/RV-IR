

#include <torch-mlir/Dialect/RISCV/IR/RISCVDialect.h>
#include <mlir/Pass/Pass.h>

namespace rair {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createLowerLinalgToRAIRPass();
} // namespace rair