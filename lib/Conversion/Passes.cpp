//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/Passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#endif // TORCH_MLIR_ENABLE_STABLEHLO

#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"


#ifdef TORCH_MLIR_ENABLE_TOSA
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#endif // TORCH_MLIR_ENABLE_TOSA

#include "torch-mlir/Conversion/LinalgToRISCV/LinalgToRISCV.h"
#include "torch-mlir/Conversion/RISCVToAffine/RISCVToAffine.h"
#include "torch-mlir/Conversion/RISCVToLLVM/RISCVToLLVM.h"
#include "torch-mlir/Conversion/RISCVToCIM/RISCVToCIM.h"
#include "torch-mlir/Conversion/CIMToLLVM/CIMToLLVM.h"

#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.h"
#include "torch-mlir/Dialect/CIM/IR/CIMOps.h"

#include "torch-mlir/Conversion/RISCVPasses.h"


//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/RISCVPasses.h.inc"
#define GEN_PASS_REGISTRATION
#include "torch-mlir/Conversion/Passes.h.inc"
#define GEN_PASS_DEF_LINALGBUFFERIZE
#include "torch-mlir/Conversion/Passes.h.inc"


} // end namespace

namespace rocc {
void registerROCCConversionPasses() {
  // 注册RISCV相关的转换Pass
  registerConvertLinalgToROCCPass();  
  registerConvertROCCToAffinePass(); 
  registerConvertROCCToLLVMPass();   
}
} // namespace rocc


namespace cim {
void registerCIMConversionPasses() {  
  registerConvertROCCToCIMPass();  
  registerConvertCIMToLLVMPass();  
}
} // namespace cim

void mlir::torch::registerConversionPasses() { 
    rocc::registerROCCConversionPasses();  
    cim::registerCIMConversionPasses();  
    ::registerPasses(); 
}


