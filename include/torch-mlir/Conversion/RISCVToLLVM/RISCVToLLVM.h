#include <torch-mlir/Dialect/RISCV/IR/RISCVDialect.h>
#include <mlir/Pass/Pass.h>

namespace rair {
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    // std::unique_ptr<mlir::Pass> createLowerLinalgToRAIRPass();
} // namespace rair