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



#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Linalg to RISCV Lowering Patterns
// @brief: Pattern to lower linalg.matmul to riscv.matmul
//===----------------------------------------------------------------------===//
struct LinalgMatmulToRISCVMatmul : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and output operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Create riscv.matmul operation
    auto matmulOp = rewriter.create<riscv::MatmulOp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg.matmul with riscv.matmul
    rewriter.replaceOp(op, matmulOp.getResult());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Linalg to RISCV Lowering Patterns
// @brief: Pattern to lower linalg.matvec to riscv.matvec
//===----------------------------------------------------------------------===//
struct LinalgMatvecToRISCVMatvec : public OpConversionPattern<linalg::MatvecOp> {
  using OpConversionPattern<linalg::MatvecOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatvecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and output operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Create riscv.matvec operation
    auto matvecOp = rewriter.create<riscv::MatvecOp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg.matvec with riscv.matvec
    rewriter.replaceOp(op, matvecOp.getResult());
    return success();
  }
};
// Pattern to lower linalg.reduce to riscv.reduce

struct LinalgReduceToRISCVReduce : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInputs()[0];
    Value outputInit = adaptor.getInits()[0];  

    auto targetOutputType = dyn_cast<RankedTensorType>(outputInit.getType());
    if (!targetOutputType)
      return rewriter.notifyMatchFailure(op, "output is not a tensor");

    SmallVector<int64_t, 4> dims;
    for (int64_t d : op.getDimensions())
      dims.push_back(d);

    StringRef kind = "sum";
    Region &region = op.getRegion(); 
    if (!region.empty()) {
      Block &body = region.front();
      for (auto &op : body) {
        if (isa<arith::AddFOp>(&op)) {
          kind = "sum";
          break;
        } else if (isa<arith::MaxNumFOp>(&op)) {
          kind = "max";
          break;
        } else if (isa<arith::MinNumFOp>(&op)) {
          kind = "min";
          break;
        }
      }
    }

    auto reduceOp = rewriter.create<riscv::ReduceOp>(
        op.getLoc(),          
        targetOutputType,     
        input,                
        dims,                 
        kind                  
    );

    rewriter.replaceOp(op, reduceOp.getResult());
    return success();
  }
};
// Pattern to lower linalg.conv2d to riscv.conv2d

struct LinalgConv2DToRISCVConv2D : public OpConversionPattern<linalg::Conv2DOp> {
  using OpConversionPattern<linalg::Conv2DOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and output operands
    Value input = adaptor.getInputs()[0];
    Value kernel = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Create riscv.conv2d operation
    auto conv2dOp = rewriter.create<riscv::Conv2DOp>(op.getLoc(), outputType, input, kernel);
    
    // Replace the linalg.conv2d with riscv.conv2d
    rewriter.replaceOp(op, conv2dOp.getResult());
    return success();
  }
};


// Pattern to lower linalg.transpose to riscv.transpose
struct LinalgTransposeToRISCVTranspose : public OpConversionPattern<linalg::TransposeOp> {
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input tensor
    Value input = adaptor.getInput();
    
    // Get the output type
    auto outputType = dyn_cast<RankedTensorType>(op.getInit().getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Check if it's a simple 2D transpose (permutation is [1, 0])
    auto permutation = op.getPermutation();
    if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0) {
      return rewriter.notifyMatchFailure(op, "only 2D transpose with [1, 0] permutation is supported");
    }
    
    // Create riscv.transpose operation
    auto transposeOp = rewriter.create<riscv::TransposeOp>(op.getLoc(), outputType, input);
    
    // Replace the linalg.transpose with riscv.transpose
    rewriter.replaceOp(op, transposeOp.getResult());
    return success();
  }
};

// Pattern to lower linalg.elemwise_binary to riscv binary ops
template <typename LinalgOp, typename RISCVOp>
struct LinalgBinaryToRISCVBinary : public OpConversionPattern<LinalgOp> {
  using OpConversionPattern<LinalgOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LinalgOp op, typename LinalgOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Create the corresponding RISCV operation
    auto riscvOp = rewriter.create<RISCVOp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg operation with RISCV operation
    rewriter.replaceOp(op, riscvOp.getResult());
    return success();
  }
};
// Pattern to lower linalg elemwise (single operand) to  unary ops
template <typename LinalgOp, typename RISCVOp>  
struct LinalgUnaryToRISCVUnary : public OpConversionPattern<LinalgOp> {
  using OpConversionPattern<LinalgOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LinalgOp op, typename LinalgOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInputs()[0];
    Value output = adaptor.getOutputs()[0];
    auto outputTensorType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputTensorType) {
      return rewriter.notifyMatchFailure(
          op, "linalg unary op output is not a RankedTensorType (expected tensor)");
    }
    auto riscvNotOp = rewriter.create<RISCVOp>(
        op.getLoc(),        
        outputTensorType,   
        input               
    );
    rewriter.replaceOp(op, riscvNotOp.getResult());

    return success();
  }
};
// Pattern to lower linalg.add to riscv.addf/addi
struct LinalgAddToRISCVAdd : public OpConversionPattern<linalg::AddOp> {
  using OpConversionPattern<linalg::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Check element type
    // auto elementType = outputType.getElementType();
    // if (isa<FloatType>(elementType)) {
    //   auto addFOp = rewriter.create<riscv::AddFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto addIOp = rewriter.create<riscv::AddIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    
      auto addOp = rewriter.create<riscv::AddOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, addOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.sub to riscv.subf/subi
struct LinalgSubToRISCVSub : public OpConversionPattern<linalg::SubOp> {
  using OpConversionPattern<linalg::SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Check element type
    // auto elementType = outputType.getElementType();
    // if (isa<FloatType>(elementType)) {
    //   auto subFOp = rewriter.create<riscv::SubFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto subIOp = rewriter.create<riscv::SubIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto subOp = rewriter.create<riscv::SubOp>(op.getLoc(), outputType, lhs, rhs);
    rewriter.replaceOp(op, subOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.mul to riscv.mulf/muli
struct LinalgMulToRISCVMul : public OpConversionPattern<linalg::MulOp> {
  using OpConversionPattern<linalg::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Check element type
    // auto elementType = outputType.getElementType();
    // if (isa<FloatType>(elementType)) {
    //   auto mulFOp = rewriter.create<riscv::MulFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto mulIOp = rewriter.create<riscv::MulIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto mulOp = rewriter.create<riscv::MulOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, mulOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.div to riscv.divf/divsi/divui
struct LinalgDivToRISCVDiv : public OpConversionPattern<linalg::DivOp> {
  using OpConversionPattern<linalg::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Check element type
    // auto elementType = outputType.getElementType();
    // if (isa<FloatType>(elementType)) {
    //   auto divFOp = rewriter.create<riscv::DivFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, divFOp.getResult());
    // } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    //   if (intType.isSigned()) {
    //     auto divSIOp = rewriter.create<riscv::DivSIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divSIOp.getResult());
    //   } else {
    //     auto divUIOp = rewriter.create<riscv::DivUIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divUIOp.getResult());
    //   }
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto divOp = rewriter.create<riscv::DivOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, divOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.negf to riscv.negf
struct LinalgNegFToRISCVNegF : public OpConversionPattern<linalg::NegFOp> {
  using OpConversionPattern<linalg::NegFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value input = adaptor.getInputs()[0];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    auto negfOp = rewriter.create<riscv::NegFOp>(op.getLoc(), outputType, input);
      rewriter.replaceOp(op, negfOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.max to riscv.max
struct LinalgMaxToRISCVMax : public OpConversionPattern<linalg::MaxOp> {
  using OpConversionPattern<linalg::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];;
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    auto maxOp = rewriter.create<riscv::MaxOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.min to riscv.min
struct LinalgMinToRISCVMin : public OpConversionPattern<linalg::MinOp> {
  using OpConversionPattern<linalg::MinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];;
    
    // Check if the output is a tensor
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    auto maxOp = rewriter.create<riscv::MinOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LinalgToRISCVLowerPass
    : public mlir::PassWrapper<LinalgToRISCVLowerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-linalg-to-riscv"; 
  }
  StringRef getDescription() const final {
    return "Lower Linalg dialect operations to RISCV dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToRISCVLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<riscv::RISCVDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void LinalgToRISCVLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // Mark RISCV dialect as legal
  target.addLegalDialect<riscv::RISCVDialect>();
  
  // Keep other dialects legal for operations we don't convert
  target.addLegalDialect<mlir::BuiltinDialect,
                         mlir::func::FuncDialect, 
                         mlir::arith::ArithDialect,
                         mlir::tensor::TensorDialect>();
  
  // Mark specific Linalg ops as illegal if they can be converted to RISCV
  target.addIllegalOp<linalg::MatmulOp>();
  target.addIllegalOp<linalg::MatvecOp>();
  target.addIllegalOp<linalg::ReduceOp>();
  target.addIllegalOp<linalg::Conv2DOp>();
  target.addIllegalOp<linalg::TransposeOp>();
  target.addIllegalOp<linalg::AddOp>();
  target.addIllegalOp<linalg::SubOp>();
  target.addIllegalOp<linalg::MulOp>();
  target.addIllegalOp<linalg::DivOp>();
  target.addIllegalOp<linalg::NegFOp>();
  target.addIllegalOp<linalg::MaxOp>();
  target.addIllegalOp<linalg::MinOp>();

  
  // Keep other Linalg ops legal (they will not be converted)
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addDynamicallyLegalDialect<mlir::linalg::LinalgDialect>(
      [](Operation *op) {
        // These specific ops are illegal, all others are legal
        return !isa<linalg::MatmulOp, linalg::MatvecOp,
                    linalg::TransposeOp, linalg::ReduceOp, 
                    linalg::AddOp, linalg::SubOp,
                    linalg::MulOp, linalg::DivOp, 
                    linalg::MaxOp,linalg::MinOp,
                    linalg::NegFOp, linalg::Conv2DOp>(op);
      });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LinalgMatmulToRISCVMatmul,
               LinalgMatvecToRISCVMatvec,
               LinalgReduceToRISCVReduce,
               LinalgConv2DToRISCVConv2D,
               LinalgTransposeToRISCVTranspose,
               LinalgAddToRISCVAdd,
               LinalgSubToRISCVSub,
               LinalgMulToRISCVMul,
               LinalgDivToRISCVDiv,
               LinalgNegFToRISCVNegF,
               LinalgMaxToRISCVMax,
               LinalgMinToRISCVMin>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace riscv{
  std::unique_ptr<mlir::Pass> createLowerLinalgToRISCVPass() {
    return std::make_unique<LinalgToRISCVLowerPass>();
  }
}
