
#ifndef RISCV_CONVERSION_PASSES_H
#define RISCV_CONVERSION_PASSES_H

//RISCV
// #include "torch-mlir/Conversion/LinalgToRISCV/LinalgToRISCV.h"
// #include "torch-mlir/Conversion/RISCVToAffine/RISCVToAffine.h"
// #include "torch-mlir/Conversion/RISCVToLLVM/RISCVToLLVM.h"
namespace rocc {

//===- Generated passes ---------------------------------------------------===//
// #define GEN_PASS_REGISTRATION
// #include "torch-mlir/Conversion/RISCVPasses.h.inc"
void registerROCCConversionPasses();
//===----------------------------------------------------------------------===//

} // namespace rocc
namespace cim {
void registerCIMConversionPasses();

} // namespace cim

#endif // RISCV_CONVERSION_PASSES_H
