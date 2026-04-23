

#include <torch-mlir/Dialect/RISCV/IR/RISCVDialect.h>
#include <mlir/Pass/Pass.h>

namespace rocc {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createLowerLinalgToROCCPass();
} // namespace rocc