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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/Sequence.h"


using namespace mlir;

//===----------------------------------------------------------------------===//
// Linalg to RISCV Lowering Patterns
// @brief: Pattern to lower linalg.matmul to rocc.matmul
//===----------------------------------------------------------------------===//
struct LinalgMatmulToROCCMatmul : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and output operands
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];
    
    // Check if the output is a tensor
    // auto outputType = dyn_cast<RankedTensorType>(output.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(op, "output is not a tensor");
    }
    
    // Create rocc.matmul operation
    // auto matmulOp = rewriter.create<rocc::MatmulOp>(op.getLoc(), outputType, lhs, rhs);
    auto matmulOp = rewriter.create<rocc::MatmulOp>(op.getLoc(), lhs, rhs, output);
    
    // Replace the linalg.matmul with rocc.matmul
    rewriter.replaceOp(op, matmulOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Linalg to RISCV Lowering Patterns
// @brief: Pattern to lower linalg.batch_matmul to rocc.batch_matmul
//===----------------------------------------------------------------------===//
struct LinalgBatchMatmulToROCCMatmul : public OpConversionPattern<linalg::BatchMatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the input and output operands
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    // Create rocc.batch_matmul operation
    auto batchMatmulOp = rewriter.create<rocc::BatchMatMulOp>(
        op.getLoc(),  
        lhs, rhs, 
        output               
    );
    
   rewriter.replaceOp(op, batchMatmulOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Linalg to RISCV Lowering Patterns
// @brief: Pattern to lower linalg.matvec to rocc.matvec
//===----------------------------------------------------------------------===//
struct LinalgMatvecToROCCMatvec : public OpConversionPattern<linalg::MatvecOp> {
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
    
    // Create rocc.matvec operation
    auto matvecOp = rewriter.create<rocc::MatvecOp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg.matvec with rocc.matvec
    rewriter.replaceOp(op, matvecOp.getResult());
    return success();
  }
};
// Pattern to lower linalg.reduce to rocc.reduce

struct LinalgReduceToROCCReduce : public OpConversionPattern<linalg::ReduceOp> {
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

    auto reduceOp = rewriter.create<rocc::ReduceOp>(
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
// Pattern to lower linalg.conv2d to rocc.conv2d

struct LinalgConv2DToROCCConv2D : public OpConversionPattern<linalg::Conv2DOp> {
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
    
    // Create rocc.conv2d operation
    auto conv2dOp = rewriter.create<rocc::Conv2DOp>(op.getLoc(), outputType, input, kernel);
    
    // Replace the linalg.conv2d with rocc.conv2d
    rewriter.replaceOp(op, conv2dOp.getResult());
    return success();
  }
};


// Pattern to lower linalg.transpose to rocc.transpose
// struct LinalgTransposeToROCCTranspose : public OpConversionPattern<linalg::TransposeOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       linalg::TransposeOp linalgTranspose, // 待转换的linalg.transpose Op
//       OpAdaptor adaptor,                   // 适配后的操作数（类型转换后）
//       ConversionPatternRewriter &rewriter) const override {
  
//     Value input = linalgTranspose.getInput();
//     Value init = linalgTranspose.getInit();
//     auto inputMemRefType = dyn_cast<MemRefType>(input.getType());
//     auto initMemRefType = dyn_cast<MemRefType>(init.getType());
//     auto permutation = linalgTranspose.getPermutation();
//     if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0) {
//       return rewriter.notifyMatchFailure(linalgTranspose, "permutation must be [1, 0]");
//     }

//     // 创建 rocc.transpose Op，替换原 linalg.transpose
//     // 注意：缓冲化后的 linalg.transpose 无返回值，直接原地修改 init
//     auto transposeOp = rewriter.create<rocc::TransposeOp>(
//         linalgTranspose.getLoc(),    // 复用原 Op 的位置信息
//         // initMemRefType,              // 返回值类型（和 init 一致）
//         input,                       // 输入 memref
//         init,                        // 输出 memref
//         rewriter.getDenseI64ArrayAttr(permutation) // permutation 属性
//     );
//     // rewriter.replaceOp(linalgTranspose, transposeOp.getResult());
//     rewriter.eraseOp(linalgTranspose);
//     return success();
//   }
// };

// Pattern to lower linalg.transpose to rocc.transpose
struct LinalgTransposeToROCCTranspose : public OpConversionPattern<linalg::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::TransposeOp linalgTranspose,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
  
    Value input = linalgTranspose.getInput();
    Value init = linalgTranspose.getInit();
    
    if (!isa<MemRefType>(input.getType()) || !isa<MemRefType>(init.getType())) {
      return rewriter.notifyMatchFailure(linalgTranspose, "input and init must be MemRefType");
    }

    auto permutation = linalgTranspose.getPermutation();
    int64_t inputRank = cast<MemRefType>(input.getType()).getRank();
    if (permutation.size() != inputRank) {
      return rewriter.notifyMatchFailure(linalgTranspose, "permutation rank mismatch with input");
    }

    rewriter.create<rocc::TransposeOp>(
        linalgTranspose.getLoc(),
        input,
        init,
        rewriter.getDenseI64ArrayAttr(permutation)
    );

    rewriter.eraseOp(linalgTranspose);
    return success();
  }
};

// struct LinalgTransposeToROCCTranspose : public OpConversionPattern<linalg::TransposeOp> {
//   using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     // Get the input tensor
//     Value input = adaptor.getInput();
//     // Get the output type
//     auto outputType = dyn_cast<RankedTensorType>(op.getInit().getType());
//     // if (!outputType) {
//     //   return rewriter.notifyMatchFailure(op, "output is not a tensor");
//     // }
//     // Check if it's a simple 2D transpose (permutation is [1, 0])
//     // auto permutation = op.getPermutation();
//     // if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0) {
//     //   return rewriter.notifyMatchFailure(op, "only 2D transpose with [1, 0] permutation is supported");
//     // }

//     SmallVector<int64_t> transpValues = {1, 0};
//     auto transpAttr = rewriter.getI64ArrayAttr(transpValues);
//     auto transposeOp = rewriter.create<rocc::TransposeOp>(
//         op.getLoc(),          // 位置信息
//         outputType,           // 输出张量类型
//         input,                // 输入张量
//         transpAttr            // transp属性
//     );
    
//     // Replace the linalg.transpose with rocc.transpose
//     rewriter.replaceOp(op, transposeOp.getResult());
//     return success();
//   }

// };


// Pattern to lower linalg.elemwise_binary to riscv binary ops
template <typename LinalgOp, typename ROCCOp>
struct LinalgBinaryToROCCBinary : public OpConversionPattern<LinalgOp> {
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
    auto roccOp = rewriter.create<ROCCOp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg operation with RISCV operation
    rewriter.replaceOp(op, roccOp.getResult());
    return success();
  }
};
// Pattern to lower linalg elemwise (single operand) to  unary ops
template <typename LinalgOp, typename ROCCOp>  
struct LinalgUnaryToROCCUnary : public OpConversionPattern<LinalgOp> {
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
    auto roccNotOp = rewriter.create<ROCCOp>(
        op.getLoc(),        
        outputTensorType,   
        input               
    );
    rewriter.replaceOp(op, roccNotOp.getResult());

    return success();
  }
};
// Pattern to lower linalg.add to rocc.addf/addi
struct LinalgAddToROCCAdd : public OpConversionPattern<linalg::AddOp> {
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
    //   auto addFOp = rewriter.create<rocc::AddFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto addIOp = rewriter.create<rocc::AddIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    
      auto addOp = rewriter.create<rocc::AddOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, addOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.sub to rocc.subf/subi
struct LinalgSubToROCCSub : public OpConversionPattern<linalg::SubOp> {
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
    //   auto subFOp = rewriter.create<rocc::SubFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto subIOp = rewriter.create<rocc::SubIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto subOp = rewriter.create<rocc::SubOp>(op.getLoc(), outputType, lhs, rhs);
    rewriter.replaceOp(op, subOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.mul to rocc.mulf/muli
struct LinalgMulToROCCMul : public OpConversionPattern<linalg::MulOp> {
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
    //   auto mulFOp = rewriter.create<rocc::MulFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto mulIOp = rewriter.create<rocc::MulIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto mulOp = rewriter.create<rocc::MulOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, mulOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.div to rocc.divf/divsi/divui
struct LinalgDivToROCCDiv : public OpConversionPattern<linalg::DivOp> {
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
    //   auto divFOp = rewriter.create<rocc::DivFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, divFOp.getResult());
    // } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    //   if (intType.isSigned()) {
    //     auto divSIOp = rewriter.create<rocc::DivSIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divSIOp.getResult());
    //   } else {
    //     auto divUIOp = rewriter.create<rocc::DivUIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divUIOp.getResult());
    //   }
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto divOp = rewriter.create<rocc::DivOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, divOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.negf to rocc.negf
struct LinalgNegFToROCCNegF : public OpConversionPattern<linalg::NegFOp> {
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
    
    auto negfOp = rewriter.create<rocc::NegFOp>(op.getLoc(), outputType, input);
      rewriter.replaceOp(op, negfOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.max to rocc.max
struct LinalgMaxToROCCMax : public OpConversionPattern<linalg::MaxOp> {
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
    
    auto maxOp = rewriter.create<rocc::MaxOp>(op.getLoc(), outputType, lhs, rhs);
      rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.min to rocc.min
struct LinalgMinToROCCMin : public OpConversionPattern<linalg::MinOp> {
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
    
    auto maxOp = rewriter.create<rocc::MinOp>(op.getLoc(), outputType, lhs, rhs);
    rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.pooling_nchw_max to rocc.pooling_nchw_max
struct LinalgPoolingNchwMaxToROCC
    : public OpConversionPattern<linalg::PoolingNchwMaxOp> {
  using OpConversionPattern<
      linalg::PoolingNchwMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::PoolingNchwMaxOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInputs()[0];
    Value kernel = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    auto dilations =
        op->getAttrOfType<TypedAttr>("dilations");
    auto strides =
        op->getAttrOfType<TypedAttr>("strides");

    auto newOp = rewriter.create<rocc::PoolingNchwMaxOp>(
        op.getLoc(),
        TypeRange{},
        ValueRange{input, kernel, output},
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("dilations", dilations),
            rewriter.getNamedAttr("strides", strides)});

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LinalgToROCCLowerPass
    : public mlir::PassWrapper<LinalgToROCCLowerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-linalg-to-rocc"; 
  }
  StringRef getDescription() const final {
    return "Lower Linalg dialect operations to ROCC dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToROCCLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<rocc::ROCCDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,scf::SCFDialect,
                    bufferization::BufferizationDialect,func::FuncDialect,
                     mlir::arith::ArithDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void LinalgToROCCLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // Mark ROCC dialect as legal
  target.addLegalDialect<rocc::ROCCDialect>();
  
  // Keep other dialects legal for operations we don't convert
  target.addLegalDialect<mlir::BuiltinDialect,
                         mlir::func::FuncDialect, 
                         mlir::arith::ArithDialect,
                         mlir::tensor::TensorDialect>();

  // Mark specific Linalg ops as illegal if they can be converted to ROCC
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
  target.addIllegalOp<linalg::BatchMatmulOp>();
  target.addIllegalOp<linalg::PoolingNchwMaxOp>();
  
  // Keep other Linalg ops legal (they will not be converted)
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  
  target.addDynamicallyLegalDialect<mlir::linalg::LinalgDialect>(
      [](Operation *op) {
        // These specific ops are illegal, all others are legal
        return !isa<linalg::MatmulOp, linalg::MatvecOp,
                    linalg::TransposeOp, linalg::ReduceOp, 
                    linalg::AddOp, linalg::SubOp,
                    linalg::MulOp, linalg::DivOp, 
                    linalg::MaxOp, linalg::MinOp,
                    linalg::NegFOp, linalg::Conv2DOp>(op);
      });

  mlir::RewritePatternSet patterns(&getContext());
  // mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
  
  patterns.add<LinalgMatmulToROCCMatmul,
               LinalgMatvecToROCCMatvec,
               LinalgReduceToROCCReduce,
               LinalgConv2DToROCCConv2D,
               LinalgTransposeToROCCTranspose,
               LinalgAddToROCCAdd,
               LinalgSubToROCCSub,
               LinalgMulToROCCMul,
               LinalgDivToROCCDiv,
               LinalgNegFToROCCNegF,
               LinalgMaxToROCCMax,
               LinalgMinToROCCMin,
               LinalgBatchMatmulToROCCMatmul,
               LinalgPoolingNchwMaxToROCC>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace rocc{
  std::unique_ptr<mlir::Pass> createLowerLinalgToROCCPass() {
    return std::make_unique<LinalgToROCCLowerPass>();
  }
}

