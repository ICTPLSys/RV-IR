

#include <torch-mlir/Dialect/RAIR/IR/RAIRDialect.h>
#include <mlir/Pass/Pass.h>

namespace rair {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createLowerLinalgToRAIRPass();
} // namespace rair