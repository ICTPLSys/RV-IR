

#include <torch-mlir/Dialect/RISCV/IR/RISCVDialect.h>
#include <mlir/Pass/Pass.h>

namespace riscv {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createLowerLinalgToRISCVPass();
} // namespace riscv