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

#include "torch-mlir/Dialect/RAIR/IR/RAIRDialect.h"
#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"
#include "torch-mlir/Conversion/RAIRPasses.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/Sequence.h"


using namespace mlir;

//===----------------------------------------------------------------------===//
// Linalg to RAIR Lowering Patterns
// @brief: Pattern to lower linalg.matmul to rair.matmul
//===----------------------------------------------------------------------===//
struct LinalgMatmulToRAIRMatmul : public OpConversionPattern<linalg::MatmulOp> {
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
    
    // Create rair.matmul operation
    // auto matmulOp = rewriter.create<rair::MatmulOp>(op.getLoc(), outputType, lhs, rhs);
    auto matmulOp = rewriter.create<rair::MatmulOp>(op.getLoc(), lhs, rhs, output, /*accelerator=*/mlir::StringAttr(), /*tile_size=*/mlir::DenseI64ArrayAttr(), /*dataflow=*/mlir::StringAttr());
    
    // Replace the linalg.matmul with rair.matmul
    rewriter.replaceOp(op, matmulOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Linalg to RAIR Lowering Patterns
// @brief: Pattern to lower linalg.batch_matmul to rair.batch_matmul
//===----------------------------------------------------------------------===//
struct LinalgBatchMatmulToRAIRMatmul : public OpConversionPattern<linalg::BatchMatmulOp> {
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
    // Create rair.batch_matmul operation
    auto batchMatmulOp = rewriter.create<rair::BatchMatMulOp>(
        op.getLoc(),
        lhs, rhs,
        output,
        /*accelerator=*/mlir::StringAttr(),
        /*tile_size=*/mlir::DenseI64ArrayAttr(),
        /*dataflow=*/mlir::StringAttr()
    );
    
   rewriter.replaceOp(op, batchMatmulOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// Linalg to RAIR Lowering Patterns
// @brief: Pattern to lower linalg.matvec to rair.matvec
//===----------------------------------------------------------------------===//
struct LinalgMatvecToRAIRMatvec : public OpConversionPattern<linalg::MatvecOp> {
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
    
    // Create rair.matvec operation
    auto matvecOp = rewriter.create<rair::MatvecOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr(), /*tile_size=*/mlir::DenseI64ArrayAttr());
    
    // Replace the linalg.matvec with rair.matvec
    rewriter.replaceOp(op, matvecOp.getResult());
    return success();
  }
};
// Pattern to lower linalg.reduce to rair.reduce

struct LinalgReduceToRAIRReduce : public OpConversionPattern<linalg::ReduceOp> {
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

    auto reduceOp = rewriter.create<rair::ReduceOp>(
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
// Pattern to lower linalg.conv2d to rair.conv2d

struct LinalgConv2DToRAIRConv2D : public OpConversionPattern<linalg::Conv2DOp> {
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
    
    // Create rair.conv2d operation
    auto conv2dOp = rewriter.create<rair::Conv2DOp>(op.getLoc(), outputType, input, kernel, /*accelerator=*/mlir::StringAttr(), /*tile_size=*/mlir::DenseI64ArrayAttr(), /*dataflow=*/mlir::StringAttr());
    
    // Replace the linalg.conv2d with rair.conv2d
    rewriter.replaceOp(op, conv2dOp.getResult());
    return success();
  }
};


// Pattern to lower linalg.transpose to rair.transpose
// struct LinalgTransposeToRAIRTranspose : public OpConversionPattern<linalg::TransposeOp> {
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

//     // 创建 rair.transpose Op，替换原 linalg.transpose
//     // 注意：缓冲化后的 linalg.transpose 无返回值，直接原地修改 init
//     auto transposeOp = rewriter.create<rair::TransposeOp>(
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

// Pattern to lower linalg.transpose to rair.transpose
struct LinalgTransposeToRAIRTranspose : public OpConversionPattern<linalg::TransposeOp> {
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

    rewriter.create<rair::TransposeOp>(
        linalgTranspose.getLoc(),
        input,
        init,
        rewriter.getDenseI64ArrayAttr(permutation),
        /*accelerator=*/mlir::StringAttr()
    );

    rewriter.eraseOp(linalgTranspose);
    return success();
  }
};

// struct LinalgTransposeToRAIRTranspose : public OpConversionPattern<linalg::TransposeOp> {
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
//     auto transposeOp = rewriter.create<rair::TransposeOp>(
//         op.getLoc(),          // 位置信息
//         outputType,           // 输出张量类型
//         input,                // 输入张量
//         transpAttr            // transp属性
//     );
    
//     // Replace the linalg.transpose with rair.transpose
//     rewriter.replaceOp(op, transposeOp.getResult());
//     return success();
//   }

// };


// Pattern to lower linalg.elemwise_binary to rair binary ops
template <typename LinalgOp, typename RAIROp>
struct LinalgBinaryToRAIRBinary : public OpConversionPattern<LinalgOp> {
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
    
    // Create the corresponding RAIR operation
    auto rairOp = rewriter.create<RAIROp>(op.getLoc(), outputType, lhs, rhs);
    
    // Replace the linalg operation with RAIR operation
    rewriter.replaceOp(op, rairOp.getResult());
    return success();
  }
};
// Pattern to lower linalg elemwise (single operand) to  unary ops
template <typename LinalgOp, typename RAIROp>  
struct LinalgUnaryToRAIRUnary : public OpConversionPattern<LinalgOp> {
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
    auto rairNotOp = rewriter.create<RAIROp>(
        op.getLoc(),        
        outputTensorType,   
        input               
    );
    rewriter.replaceOp(op, rairNotOp.getResult());

    return success();
  }
};
// Pattern to lower linalg.add to rair.addf/addi
struct LinalgAddToRAIRAdd : public OpConversionPattern<linalg::AddOp> {
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
    //   auto addFOp = rewriter.create<rair::AddFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto addIOp = rewriter.create<rair::AddIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, addIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    
      auto addOp = rewriter.create<rair::AddOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
      rewriter.replaceOp(op, addOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.sub to rair.subf/subi
struct LinalgSubToRAIRSub : public OpConversionPattern<linalg::SubOp> {
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
    //   auto subFOp = rewriter.create<rair::SubFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto subIOp = rewriter.create<rair::SubIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, subIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto subOp = rewriter.create<rair::SubOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
    rewriter.replaceOp(op, subOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.mul to rair.mulf/muli
struct LinalgMulToRAIRMul : public OpConversionPattern<linalg::MulOp> {
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
    //   auto mulFOp = rewriter.create<rair::MulFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulFOp.getResult());
    // } else if (isa<IntegerType>(elementType)) {
    //   auto mulIOp = rewriter.create<rair::MulIOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, mulIOp.getResult());
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto mulOp = rewriter.create<rair::MulOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
      rewriter.replaceOp(op, mulOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.div to rair.divf/divsi/divui
struct LinalgDivToRAIRDiv : public OpConversionPattern<linalg::DivOp> {
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
    //   auto divFOp = rewriter.create<rair::DivFOp>(op.getLoc(), outputType, lhs, rhs);
    //   rewriter.replaceOp(op, divFOp.getResult());
    // } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    //   if (intType.isSigned()) {
    //     auto divSIOp = rewriter.create<rair::DivSIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divSIOp.getResult());
    //   } else {
    //     auto divUIOp = rewriter.create<rair::DivUIOp>(op.getLoc(), outputType, lhs, rhs);
    //     rewriter.replaceOp(op, divUIOp.getResult());
    //   }
    // } else {
    //   return rewriter.notifyMatchFailure(op, "unsupported element type");
    // }
    auto divOp = rewriter.create<rair::DivOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
      rewriter.replaceOp(op, divOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.negf to rair.negf
struct LinalgNegFToRAIRNegF : public OpConversionPattern<linalg::NegFOp> {
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
    
    auto negfOp = rewriter.create<rair::NegFOp>(op.getLoc(), outputType, input);
      rewriter.replaceOp(op, negfOp.getResult());
    
    return success();
  }
};
// Pattern to lower linalg.max to rair.max
struct LinalgMaxToRAIRMax : public OpConversionPattern<linalg::MaxOp> {
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
    
    auto maxOp = rewriter.create<rair::MaxOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
      rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.min to rair.min
struct LinalgMinToRAIRMin : public OpConversionPattern<linalg::MinOp> {
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
    
    auto maxOp = rewriter.create<rair::MinOp>(op.getLoc(), outputType, lhs, rhs, /*accelerator=*/mlir::StringAttr());
    rewriter.replaceOp(op, maxOp.getResult());
    
    return success();
  }
};

// Pattern to lower linalg.pooling_nchw_max to rair.pooling_nchw_max
struct LinalgPoolingNchwMaxToRAIR
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

    auto newOp = rewriter.create<rair::PoolingNchwMaxOp>(
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

// Pattern to lower linalg.pooling_nchw_sum to rair.pooling_nchw_sum
struct LinalgPoolingNchwSumToRAIR
    : public OpConversionPattern<linalg::PoolingNchwSumOp> {
  using OpConversionPattern<
      linalg::PoolingNchwSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::PoolingNchwSumOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInputs()[0];
    Value kernel = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    auto dilations = op->getAttrOfType<TypedAttr>("dilations");
    auto strides = op->getAttrOfType<TypedAttr>("strides");

    auto newOp = rewriter.create<rair::PoolingNchwSumOp>(
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

// Pattern to lower linalg.conv_2d_nchw_fchw to rair.conv_2d_nchw_fchw
struct LinalgConv2DNchwFchwToRAIR
    : public OpConversionPattern<linalg::Conv2DNchwFchwOp> {
  using OpConversionPattern<
      linalg::Conv2DNchwFchwOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::Conv2DNchwFchwOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.getInputs()[0];
    Value kernel = adaptor.getInputs()[1];
    Value output = adaptor.getOutputs()[0];

    auto dilations = op->getAttrOfType<TypedAttr>("dilations");
    auto strides = op->getAttrOfType<TypedAttr>("strides");

    auto newOp = rewriter.create<rair::Conv2dNchwFchwOp>(
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
// MemRef to RAIR Lowering Patterns
//===----------------------------------------------------------------------===//
struct MemRefAllocToRAIRAlloc : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto allocOp = rewriter.create<rair::AllocOp>(op.getLoc(), op.getType());
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct MemRefCopyToRAIRTransfer : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<rair::TransferOp>(
        op.getLoc(),
        adaptor.getSource(), adaptor.getTarget(),
        /*src_memory_space=*/rair::MemorySpaceAttr(),
        /*dst_memory_space=*/rair::MemorySpaceAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LinalgToRAIRLowerPass
    : public mlir::PassWrapper<LinalgToRAIRLowerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-linalg-to-rair"; 
  }
  StringRef getDescription() const final {
    return "Lower Linalg dialect operations to RAIR dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToRAIRLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<rair::RAIRDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,scf::SCFDialect,
                    bufferization::BufferizationDialect,func::FuncDialect,
                     mlir::arith::ArithDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void LinalgToRAIRLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // Mark RAIR dialect as legal
  target.addLegalDialect<rair::RAIRDialect>();

  // Keep other dialects legal for operations we don't convert
  target.addLegalDialect<mlir::BuiltinDialect,
                         mlir::func::FuncDialect,
                         mlir::arith::ArithDialect,
                         mlir::tensor::TensorDialect>();

  // Mark specific Linalg ops as illegal if they can be converted to RAIR
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
  target.addIllegalOp<linalg::PoolingNchwSumOp>();
  target.addIllegalOp<linalg::Conv2DNchwFchwOp>();
  target.addIllegalOp<memref::AllocOp>();
  target.addIllegalOp<memref::CopyOp>();

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
                    linalg::NegFOp, linalg::Conv2DOp,
                    linalg::BatchMatmulOp, linalg::PoolingNchwMaxOp,
                    linalg::PoolingNchwSumOp,
                    linalg::Conv2DNchwFchwOp>(op);
      });

  target.addDynamicallyLegalDialect<mlir::memref::MemRefDialect>(
      [](Operation *op) {
        return !isa<memref::AllocOp, memref::CopyOp>(op);
      });

  mlir::RewritePatternSet patterns(&getContext());
  // mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);

  patterns.add<LinalgMatmulToRAIRMatmul,
               LinalgMatvecToRAIRMatvec,
               LinalgReduceToRAIRReduce,
               LinalgConv2DToRAIRConv2D,
               LinalgTransposeToRAIRTranspose,
               LinalgAddToRAIRAdd,
               LinalgSubToRAIRSub,
               LinalgMulToRAIRMul,
               LinalgDivToRAIRDiv,
               LinalgNegFToRAIRNegF,
               LinalgMaxToRAIRMax,
               LinalgMinToRAIRMin,
               LinalgBatchMatmulToRAIRMatmul,
               LinalgPoolingNchwMaxToRAIR,
               LinalgPoolingNchwSumToRAIR,
               LinalgConv2DNchwFchwToRAIR,
               MemRefAllocToRAIRAlloc,
               MemRefCopyToRAIRTransfer>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Insert accelerator context management (acquire/release) and local memory
  // (SRAM) data movement for functions that contain RAIR compute ops.
  // NPU programming model: data must be explicitly moved from Global Memory
  // (DRAM) to Local Memory (SRAM) before computation, and results moved back.
  getOperation()->walk([&](func::FuncOp funcOp) {
    if (funcOp.isExternal() || funcOp.getBody().empty())
      return;

    // Check if this function has any RAIR compute ops
    bool hasRAIROps = false;
    funcOp.walk([&](Operation *op) {
      if (isa<rair::MatmulOp, rair::BatchMatMulOp, rair::MatvecOp,
              rair::Conv2DOp, rair::Conv2dNchwFchwOp,
              rair::PoolingNchwMaxOp, rair::PoolingNchwSumOp,
              rair::ReduceOp, rair::TransposeOp,
              rair::AddOp, rair::SubOp, rair::MulOp, rair::DivOp,
              rair::MaxOp, rair::MinOp, rair::NegFOp>(op))
        hasRAIROps = true;
    });
    if (!hasRAIROps)
      return;

    OpBuilder builder(funcOp.getContext());
    Block &entry = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entry);

    // Acquire accelerator context at function entry
    auto ctxType = rair::ContextType::get(funcOp.getContext());
    auto acquireOp = builder.create<rair::AcquireOp>(
        funcOp.getLoc(), ctxType,
        builder.getStringAttr("default"));

    // Insert local memory (SRAM) data movement for matmul/batch_matmul ops.
    // For each compute op: alloc local buffers -> transfer GMEM->LMEM ->
    // compute on LMEM -> transfer LMEM->GMEM -> dealloc local buffers.
    SmallVector<Operation *> computeOps;
    funcOp.walk([&](Operation *op) {
      if (isa<rair::MatmulOp, rair::BatchMatMulOp>(op))
        computeOps.push_back(op);
    });

    for (auto *computeOp : computeOps) {
      Value lhs = computeOp->getOperand(0);
      Value rhs = computeOp->getOperand(1);
      Value output = computeOp->getOperand(2);
      auto loc = computeOp->getLoc();

      auto lhsType = cast<MemRefType>(lhs.getType());
      auto rhsType = cast<MemRefType>(rhs.getType());
      auto outputType = cast<MemRefType>(output.getType());

      auto globalMemory = rair::MemorySpaceAttr::get(
          builder.getContext(), rair::MemorySpace::GMEM);
      auto localMemory = rair::MemorySpaceAttr::get(
          builder.getContext(), rair::MemorySpace::LMEM);

      // Create contiguous local memory types (strip strided layout)
      auto lhsLocalType = MemRefType::get(
          lhsType.getShape(), lhsType.getElementType(),
          MemRefLayoutAttrInterface(), localMemory);
      auto rhsLocalType = MemRefType::get(
          rhsType.getShape(), rhsType.getElementType(),
          MemRefLayoutAttrInterface(), localMemory);
      auto outputLocalType = MemRefType::get(
          outputType.getShape(), outputType.getElementType(),
          MemRefLayoutAttrInterface(), localMemory);

      builder.setInsertionPoint(computeOp);

      // Allocate local memory (SRAM) buffers
      auto lhsLocal = builder.create<rair::AllocBufferOp>(
          loc, lhsLocalType, acquireOp.getResult(), localMemory);
      auto rhsLocal = builder.create<rair::AllocBufferOp>(
          loc, rhsLocalType, acquireOp.getResult(), localMemory);
      auto outLocal = builder.create<rair::AllocBufferOp>(
          loc, outputLocalType, acquireOp.getResult(), localMemory);

      // Transfer data from Global Memory to Local Memory
      builder.create<rair::TransferOp>(
          loc, lhs, lhsLocal.getResult(), globalMemory, localMemory);
      builder.create<rair::TransferOp>(
          loc, rhs, rhsLocal.getResult(), globalMemory, localMemory);
      builder.create<rair::TransferOp>(
          loc, output, outLocal.getResult(), globalMemory, localMemory);

      // Update compute op to use local memory buffers
      computeOp->setOperand(0, lhsLocal.getResult());
      computeOp->setOperand(1, rhsLocal.getResult());
      computeOp->setOperand(2, outLocal.getResult());

      // Transfer result from Local Memory back to Global Memory
      builder.setInsertionPointAfter(computeOp);
      builder.create<rair::TransferOp>(
          loc, outLocal.getResult(), output, localMemory, globalMemory);

      // Deallocate local memory buffers
      builder.create<rair::DeallocBufferOp>(
          loc, acquireOp.getResult(), lhsLocal.getResult());
      builder.create<rair::DeallocBufferOp>(
          loc, acquireOp.getResult(), rhsLocal.getResult());
      builder.create<rair::DeallocBufferOp>(
          loc, acquireOp.getResult(), outLocal.getResult());
    }

    // Release accelerator context before each return
    funcOp.walk([&](func::ReturnOp retOp) {
      builder.setInsertionPoint(retOp);
      builder.create<rair::ReleaseOp>(retOp.getLoc(), acquireOp.getResult());
    });
  });
}

namespace rair{
  std::unique_ptr<mlir::Pass> createLowerLinalgToRAIRPass() {
    return std::make_unique<LinalgToRAIRLowerPass>();
  }
}
