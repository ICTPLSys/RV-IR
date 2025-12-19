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

#include "torch-mlir/Dialect/RISCV/IR/RISCVDialect.h"
#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.h"
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
namespace riscv{
class PrintOpLowering : public mlir::ConversionPattern {
public:
  explicit PrintOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(riscv::PrintOp::getOperationName(), 1,
                                context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();                
    auto memRefType = mlir::cast<mlir::MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    
    // Get the element type to determine the correct format specifier
    auto elementType = memRefType.getElementType();
    std::string formatSpec;
    
    if (mlir::isa<mlir::FloatType>(elementType)) {
      auto floatType = mlir::cast<mlir::FloatType>(elementType);
      if (floatType.getWidth() == 64) {
        formatSpec = "%lf ";
      } else if (floatType.getWidth() == 32) {
        formatSpec = "%f ";
      } else if (floatType.getWidth() == 16) {
        formatSpec = "%f ";  // Float16 will be promoted to float for printf
      } else {
        formatSpec = "%f ";
      }
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      auto intType = mlir::cast<mlir::IntegerType>(elementType);
      if (intType.getWidth() == 64) {
        // MLIR integers are signless by default, use signed format
        formatSpec = "%lld ";
      } else if (intType.getWidth() == 32) {
        // MLIR integers are signless by default, use signed format
        formatSpec = "%d ";
      } else if (intType.getWidth() == 16) {
        // MLIR integers are signless by default, use signed format
        formatSpec = "%hd ";
      } else if (intType.getWidth() == 8) {
        // MLIR integers are signless by default, use signed format
        formatSpec = "%hhd ";
      } else {
        formatSpec = "%d ";
      }
    } else {
      formatSpec = "%f ";  // Default fallback
    }
    
    formatSpec += '\0';
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec_" + std::to_string(std::hash<std::string>{}(formatSpec)), 
        mlir::StringRef(formatSpec), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", mlir::StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    mlir::SmallVector<mlir::Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      for (mlir::Operation &nested : *loop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1) {
        rewriter.create<mlir::LLVM::CallOp>(loc, getPrintfType(context),
                                            printfRef, newLineCst);
      }
      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = mlir::cast<riscv::PrintOp>(op);
    auto elementLoad =
        rewriter.create<mlir::memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    
    // Convert the loaded value to a type that printf can handle
    mlir::Value printValue = elementLoad;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      auto floatType = mlir::cast<mlir::FloatType>(elementType);
      // Convert f16 to f64 for printf (via f32)
      if (floatType.getWidth() == 16) {
        auto f32Value = rewriter.create<mlir::arith::ExtFOp>(
            loc, rewriter.getF32Type(), elementLoad,
            mlir::arith::FastMathFlagsAttr());
        printValue = rewriter.create<mlir::arith::ExtFOp>(
            loc, rewriter.getF64Type(), f32Value,
            mlir::arith::FastMathFlagsAttr());
      } else if (floatType.getWidth() == 32) {
        // Convert f32 to f64 for printf (C varargs promotion)
        printValue = rewriter.create<mlir::arith::ExtFOp>(
            loc, rewriter.getF64Type(), elementLoad,
            mlir::arith::FastMathFlagsAttr());
      }
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      auto intType = mlir::cast<mlir::IntegerType>(elementType);
      // For integers smaller than 32-bit, extend them
      if (intType.getWidth() < 32) {
        // MLIR integers are signless, use signed extension
        printValue = rewriter.create<mlir::arith::ExtSIOp>(
            loc, rewriter.getI32Type(), elementLoad);
      }
    }
    
    rewriter.create<mlir::LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        mlir::ArrayRef<mlir::Value>({formatSpecifierCst, printValue}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                        /*isVarArg=*/true);
    return llvmFnType;
  }

  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(context, "printf");
    }

    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));

  }
};

class WorldOpLowering : public mlir::ConversionPattern {
public:
  explicit WorldOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(riscv::WorldOp::getOperationName(), 1,
                                context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    auto loc = op->getLoc();
    mlir::Value riscvWorld = getOrCreateGlobalString(
        loc, rewriter, "riscv_word_string",
        mlir::StringRef("RISCV, World! \n\0", 16), parentModule);

    rewriter.create<mlir::LLVM::CallOp>(loc, getPrintfType(context),
                                        printfRef, riscvWorld);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                        /*isVarArg=*/true);
    return llvmFnType;
  }

  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      return mlir::SymbolRefAttr::get(context, "printf");
    }

    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));

  }
};
} // namespace riscv

namespace {
class RISCVToLLVMLoweringPass
    : public mlir::PassWrapper<RISCVToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-riscv-to-llvm"; 
  }
  StringRef getDescription() const final {
    return "Lower RISCV dialect operations to LLVM dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RISCVToLLVMLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void RISCVToLLVMLoweringPass::runOnOperation() {
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

  patterns.add<riscv::PrintOpLowering, riscv::WorldOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

// std::unique_ptr<mlir::Pass> riscv::createLowerToLLVMPass() {
//   return std::make_unique<RISCVToLLVMLoweringPass>();
// }
namespace riscv {
  std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
    return std::make_unique<RISCVToLLVMLoweringPass>(); 
  }
}