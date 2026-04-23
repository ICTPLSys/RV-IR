#include <torch-mlir/Dialect/RISCV/IR/RISCVDialect.h>
#include <mlir/Pass/Pass.h>

namespace rocc {

    std::unique_ptr<mlir::Pass> createLowerToAffinePass();
} // namespace rocc