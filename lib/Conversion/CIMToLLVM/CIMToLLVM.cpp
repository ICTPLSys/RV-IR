// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "torch-mlir/Dialect/CIM/IR/CIMDialect.h"
#include "torch-mlir/Dialect/CIM/IR/CIMOps.h"
#include "torch-mlir/Conversion/RISCVPasses.h"
// #include "torch-mlir/Conversion/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include <iostream>
using namespace mlir;

namespace {
class CIMToLLVMLoweringPass
    : public mlir::PassWrapper<CIMToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-cim-to-llvm"; 
  }
  StringRef getDescription() const final {
    return "Lower CIM dialect operations to LLVM IR";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIMToLLVMLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<cim::CIMDialect, mlir::LLVM::LLVMDialect, 
                    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void CIMToLLVMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());
  mlir::RewritePatternSet patterns(&getContext());

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace cim {
  std::unique_ptr<mlir::Pass> createCIMLowerToLLVMPass() {
    return std::make_unique<CIMToLLVMLoweringPass>(); 
  }
}