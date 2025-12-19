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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"

using namespace mlir;

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}
static mlir::Value insertAllocAndDealloc_OpBuilder(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::OpBuilder &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}
//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//
//Scalar type
struct ScalarConstantOpLowering
    : public OpConversionPattern<riscv::ConstantOp> {
  using OpConversionPattern<riscv::ConstantOp>::OpConversionPattern;
  // mlir::LogicalResult match(riscv::ConstantOp op) const override {

  //   // if (!op.getType().isa<mlir::TensorType>()) {
  //   if (!mlir::isa<mlir::TensorType>(op.getType())) {
  //     return mlir::success();
  //   }
  //   return mlir::failure();
  // }

  LogicalResult
  matchAndRewrite(riscv::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypedAttr valueTypedAttr = op.getValue(); 
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, valueTypedAttr);
    return success();
  }
};
class TensorConstantOpLowering : public mlir::OpRewritePattern<riscv::ConstantOp> {
  using OpRewritePattern<riscv::ConstantOp>::OpRewritePattern;
    // 仅匹配Tensor类型
  // mlir::LogicalResult match(riscv::ConstantOp op) const override {
  //   // if (!op.getType().isa<mlir::TensorType>()) {
  //   if (!mlir::isa<mlir::TensorType>(op.getType())) {
  //     return mlir::success();
  //   }
  //   return mlir::failure();
  // }
  mlir::LogicalResult
  matchAndRewrite(riscv::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const final {
    //DenseElementsAttr 是 MLIR 中表示 “稠密元素属性” 的类型（可存储张量 / 向量的所有元素值
    mlir::Attribute attr = op.getValue();
    mlir::DenseElementsAttr constantValue =
        llvm::dyn_cast<mlir::DenseElementsAttr>(attr);
    if (!constantValue)
      return rewriter.notifyMatchFailure(op, "expected DenseElementsAttr");

    // mlir::DenseElementsAttr constantValue = op.getValue(); 
    mlir::Location loc = op.getLoc();
    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = mlir::cast<mlir::TensorType>(op.getType());

    auto memRefType = convertTensorToMemRef(tensorType);  //这里的类型转换对于reshape不适用
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    mlir::SmallVector<mlir::Value, 2> indices;

    mlir::Type elementType = memRefType.getElementType();
    
    // if(elementType.isa<mlir::FloatType>()){
    if (mlir::isa<mlir::FloatType>(elementType)) {
      auto valueIt = constantValue.getValues<mlir::FloatAttr>().begin();
      std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::affine::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);
       
    // }else if(elementType.isa<mlir::IntegerType>()){
    }else if(mlir::isa<mlir::IntegerType>(elementType)){
      auto valueIt = constantValue.getValues<mlir::IntegerAttr>().begin();
            std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::affine::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);
    }else{
      return rewriter.notifyMatchFailure(
      op, "unsupported element type for constant lowering");
    }


    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};
// class ConstantOpLowering : public mlir::OpRewritePattern<riscv::ConstantOp> {
//   using OpRewritePattern<riscv::ConstantOp>::OpRewritePattern;

//   mlir::LogicalResult
//   matchAndRewrite(riscv::ConstantOp op,
//                   mlir::PatternRewriter &rewriter) const final {
//     //DenseElementsAttr 是 MLIR 中表示 “稠密元素属性” 的类型（可存储张量 / 向量的所有元素值
//     mlir::Attribute attr = op.getValue();
//     mlir::DenseElementsAttr constantValue =
//         llvm::dyn_cast<mlir::DenseElementsAttr>(attr);
//     if (!constantValue)
//       return rewriter.notifyMatchFailure(op, "expected DenseElementsAttr");

//     // mlir::DenseElementsAttr constantValue = op.getValue(); 
//     mlir::Location loc = op.getLoc();
//     // When lowering the constant operation, we allocate and assign the constant
//     // values to a corresponding memref allocation.
//     auto tensorType = mlir::cast<mlir::TensorType>(op.getType());

//     auto memRefType = convertTensorToMemRef(tensorType);
//     auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

//     // We will be generating constant indices up-to the largest dimension.
//     // Create these constants up-front to avoid large amounts of redundant
//     // operations.
//     auto valueShape = memRefType.getShape();
//     mlir::SmallVector<mlir::Value, 8> constantIndices;

//     if (!valueShape.empty()) {
//       for (auto i : llvm::seq<int64_t>(
//                0, *std::max_element(valueShape.begin(), valueShape.end())))
//         constantIndices.push_back(
//             rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
//     } else {
//       // This is the case of a tensor of rank 0.
//       constantIndices.push_back(
//           rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
//     }
//     // The constant operation represents a multi-dimensional constant, so we
//     // will need to generate a store for each of the elements. The following
//     // functor recursively walks the dimensions of the constant shape,
//     // generating a store when the recursion hits the base case.

//     // [4, 3] (1, 2, 3, 4, 5, 6, 7, 8)
//     // storeElements(0)
//     //   indices = [0]
//     //   storeElements(1)
//     //     indices = [0, 0]
//     //     storeElements(2)
//     //       store (const 1) [0, 0]
//     //     indices = [0]
//     //     indices = [0, 1]
//     //     storeElements(2)
//     //       store (const 2) [0, 1]
//     //  ...
//     //
//     mlir::SmallVector<mlir::Value, 2> indices;
//     mlir::Type elementType = memRefType.getElementType();
//     if(mlir::isa<mlir::FloatType>(elementType)){
//       auto valueIt = constantValue.value_begin<mlir::APFloat>();
//       std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
//       // The last dimension is the base case of the recursion, at this point
//       // we store the element at the given index.
//       if (dimension == valueShape.size()) {
//         auto floatAttr = rewriter.getFloatAttr(elementType, *valueIt);
//         rewriter.create<mlir::affine::AffineStoreOp>(
//             loc, rewriter.create<mlir::arith::ConstantOp>(loc, floatAttr),
//             alloc, llvm::ArrayRef(indices));
//         ++valueIt;
//         return;
//       }

//       // Otherwise, iterate over the current dimension and add the indices to
//       // the list.
//       for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
//         indices.push_back(constantIndices[i]);
//         storeElements(dimension + 1);
//         indices.pop_back();
//       }
//     };

//     // Start the element storing recursion from the first dimension.
//     storeElements(/*dimension=*/0);
       
//     }else if(mlir::isa<mlir::IntegerType>(elementType)){
//       auto valueIt = constantValue.value_begin<mlir::APInt>();
//       std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
//       // The last dimension is the base case of the recursion, at this point
//       // we store the element at the given index.
//       if (dimension == valueShape.size()) {
//         auto intAttr = rewriter.getIntegerAttr(elementType, *valueIt);
//         rewriter.create<mlir::affine::AffineStoreOp>(
//             loc, rewriter.create<mlir::arith::ConstantOp>(loc, intAttr),
//             alloc, llvm::ArrayRef(indices));
//         ++valueIt;
//         return;
//       }

//       // Otherwise, iterate over the current dimension and add the indices to
//       // the list.
//       for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
//         indices.push_back(constantIndices[i]);
//         storeElements(dimension + 1);
//         indices.pop_back();
//       }
//     };

//     // Start the element storing recursion from the first dimension.
//     storeElements(/*dimension=*/0);
//     }else{
//       return rewriter.notifyMatchFailure(
//       op, "unsupported element type for constant lowering");
//     }


//     // Replace this operation with the generated alloc.
//     rewriter.replaceOp(op, alloc);
//     return mlir::success();
//   }
// };

//===----------------------------------------------------------------------===//
// ArithConstantOpLowering
//===----------------------------------------------------------------------===//
class ArithConstantOpLowering : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern<mlir::arith::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Only handle tensor constants that need conversion
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!tensorType)
      return rewriter.notifyMatchFailure(op, "not a tensor constant");

    // Get the constant value
    auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return rewriter.notifyMatchFailure(op, "not a dense elements attr");

    // Create a memref allocation
    auto memrefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memrefType, op.getLoc(), rewriter);

    // Copy the constant values into the memref
    auto shape = tensorType.getShape();
    SmallVector<int64_t, 4> indices(shape.size(), 0);
    
    std::function<void(unsigned)> storeElements = [&](unsigned dim) {
      if (dim == shape.size()) {
        // We've reached the innermost dimension, store the value
        SmallVector<Value> indexValues;
        for (auto idx : indices) {
          indexValues.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(
              op.getLoc(), idx));
        }
        
        // Get the value at the current indices
        auto flatIndex = 0;
        auto strides = SmallVector<int64_t>(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * shape[i + 1];
        }
        for (size_t i = 0; i < indices.size(); ++i) {
          flatIndex += indices[i] * strides[i];
        }
        
        auto elementAttr = denseAttr.getValues<mlir::Attribute>()[flatIndex];
        auto typedAttr = llvm::cast<mlir::TypedAttr>(elementAttr);
        auto elementValue = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), typedAttr);
        
        // Store the value
        rewriter.create<mlir::affine::AffineStoreOp>(
            op.getLoc(), elementValue, alloc, indexValues);
        return;
      }
      
      // Iterate over the current dimension
      for (int64_t i = 0; i < shape[dim]; ++i) {
        indices[dim] = i;
        storeElements(dim + 1);
      }
    };
    
    storeElements(0);
    
    // Replace the tensor constant with the memref
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// PrintOpLowering
//===----------------------------------------------------------------------===//
// TODO:"riscv.print"(%2) : (tensor<i64>) -> ()
class PrintOpLowering : public mlir::OpConversionPattern<riscv::PrintOp> {
  using OpConversionPattern<riscv::PrintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(riscv::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We don't lower "riscv.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Matmul operations
//===----------------------------------------------------------------------===//
struct MatmulOpLowering : public ConversionPattern {
  MatmulOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::MatmulOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
      auto loc = op->getLoc();
      
      // Get the operand types and shapes
      auto matmulOp = cast<riscv::MatmulOp>(op);
      auto lhsType = mlir::cast<RankedTensorType>(matmulOp.getLhs().getType());
      auto rhsType = mlir::cast<RankedTensorType>(matmulOp.getRhs().getType());
      auto resultType = mlir::cast<RankedTensorType>(matmulOp.getResult().getType());
      
      // Extract dimensions: lhs is MxK, rhs is KxN, result is MxN
      int64_t M = lhsType.getShape()[0];
      int64_t K = lhsType.getShape()[1];
      int64_t N = rhsType.getShape()[1];
      
      // Allocate result memref
      auto memRefType = convertTensorToMemRef(resultType);
      auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
      
      // Initialize result to zero based on element type
      auto elementType = resultType.getElementType();
      Value zero;
      
      if (mlir::isa<FloatType>(elementType)) {
        auto zeroAttr = rewriter.getZeroAttr(elementType);
        zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      } else if (mlir::isa<IntegerType>(elementType)) {
        auto zeroAttr = rewriter.getZeroAttr(elementType);
        zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      } else {
        return failure();
      }
      
      // Initialize result matrix to zero
      SmallVector<int64_t, 2> initLowerBounds = {0, 0};
      SmallVector<int64_t, 2> initUpperBounds = {M, N};
      SmallVector<int64_t, 2> initSteps = {1, 1};
      
      affine::buildAffineLoopNest(
          rewriter, loc, initLowerBounds, initUpperBounds, initSteps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            nestedBuilder.create<affine::AffineStoreOp>(loc, zero, alloc, ivs);
          });
      
      // Create three nested loops for matrix multiplication
      SmallVector<int64_t, 3> lowerBounds = {0, 0, 0};
      SmallVector<int64_t, 3> upperBounds = {M, N, K};
      SmallVector<int64_t, 3> steps = {1, 1, 1};
      
      affine::buildAffineLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            Value i = ivs[0]; // Row index for result and lhs
            Value j = ivs[1]; // Column index for result and rhs
            Value k = ivs[2]; // Reduction dimension
            
            // Load lhs[i][k] and rhs[k][j]
            auto loadedLhs = nestedBuilder.create<affine::AffineLoadOp>(
                loc, operands[0], ValueRange{i, k});
            auto loadedRhs = nestedBuilder.create<affine::AffineLoadOp>(
                loc, operands[1], ValueRange{k, j});
            
            // Multiply lhs[i][k] * rhs[k][j]
            Value mul;
            if (mlir::isa<FloatType>(elementType)) {
              mul = nestedBuilder.create<arith::MulFOp>(loc, loadedLhs, loadedRhs);
            } else if (mlir::isa<IntegerType>(elementType)) {
              mul = nestedBuilder.create<arith::MulIOp>(loc, loadedLhs, loadedRhs);
            }
            
            // Load current accumulator value
            auto currentSum = nestedBuilder.create<affine::AffineLoadOp>(
                loc, alloc, ValueRange{i, j});
            
            // Add to accumulator
            Value newSum;
            if (mlir::isa<FloatType>(elementType)) {
              newSum = nestedBuilder.create<arith::AddFOp>(loc, currentSum, mul);
            } else if (mlir::isa<IntegerType>(elementType)) {
              newSum = nestedBuilder.create<arith::AddIOp>(loc, currentSum, mul);
            }
            
            // Store back
            nestedBuilder.create<affine::AffineStoreOp>(loc, newSum, alloc, ValueRange{i, j});
          });
      
      // Replace operation with result allocation
      rewriter.replaceOp(op, alloc);
      return success();
    }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Matvec operations
//===----------------------------------------------------------------------===//
struct MatvecOpLowering : public ConversionPattern {
  MatvecOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::MatvecOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    
    auto matvecOp = cast<riscv::MatvecOp>(op);
    auto lhsType = mlir::cast<RankedTensorType>(matvecOp.getLhs().getType());
    auto rhsType = mlir::cast<RankedTensorType>(matvecOp.getRhs().getType());
    auto resultType = mlir::cast<RankedTensorType>(matvecOp.getResult().getType());

    int64_t M = lhsType.getShape()[0];       
    int64_t N = lhsType.getShape()[1];       
    int64_t vecLen = rhsType.getShape()[0];  

    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto elementType = resultType.getElementType();
    Value zero;
    if (mlir::isa<FloatType>(elementType)) {
      auto zeroAttr = rewriter.getZeroAttr(elementType);
      zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    } else if (mlir::isa<IntegerType>(elementType)) {
      auto zeroAttr = rewriter.getZeroAttr(elementType);
      zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    } else {
      return failure();
    }

    SmallVector<int64_t, 1> initLowerBounds = {0};    
    SmallVector<int64_t, 1> initUpperBounds = {M};    
    SmallVector<int64_t, 1> initSteps = {1};          
    affine::buildAffineLoopNest(
        rewriter, loc, initLowerBounds, initUpperBounds, initSteps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
         
          nestedBuilder.create<affine::AffineStoreOp>(loc, zero, alloc, ivs);
        });

    SmallVector<int64_t, 2> lowerBounds = {0, 0};    
    SmallVector<int64_t, 2> upperBounds = {M, N};   
    SmallVector<int64_t, 2> steps = {1, 1};         
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      
          Value i = ivs[0];
          Value k = ivs[1];
          auto loadedLhs = nestedBuilder.create<affine::AffineLoadOp>(
              loc, operands[0], ValueRange{i, k});
      
          auto loadedRhs = nestedBuilder.create<affine::AffineLoadOp>(
              loc, operands[1], ValueRange{k});

          Value mul;
          if (mlir::isa<FloatType>(elementType)) {
            mul = nestedBuilder.create<arith::MulFOp>(loc, loadedLhs, loadedRhs);
          } else if (mlir::isa<IntegerType>(elementType)) {
            mul = nestedBuilder.create<arith::MulIOp>(loc, loadedLhs, loadedRhs);
          } 

          auto currentSum = nestedBuilder.create<affine::AffineLoadOp>(
              loc, alloc, ValueRange{i});

          Value newSum;
          if (mlir::isa<FloatType>(elementType)) {
            newSum = nestedBuilder.create<arith::AddFOp>(loc, currentSum, mul);
          } else if (mlir::isa<IntegerType>(elementType)) {
            newSum = nestedBuilder.create<arith::AddIOp>(loc, currentSum, mul);
          } 
       
          nestedBuilder.create<affine::AffineStoreOp>(loc, newSum, alloc, ValueRange{i});
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Transpose operations (不支持 transp 参数)
//===----------------------------------------------------------------------===//
// struct TransposeOpLowering : public ConversionPattern {
//   TransposeOpLowering(MLIRContext *ctx)
//       : ConversionPattern(riscv::TransposeOp::getOperationName(), 1, ctx) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto loc = op->getLoc();
//     lowerOpToLoops(op, operands, rewriter,
//                    [loc](OpBuilder &builder, ValueRange memRefOperands,
//                          ValueRange loopIvs) {
//                      // Generate an adaptor for the remapped operands of the
//                      // TransposeOp. This allows for using the nice named
//                      // accessors that are generated by the ODS.
//                      riscv::TransposeOpAdaptor transposeAdaptor(memRefOperands);
//                      Value input = transposeAdaptor.getInput();

//                      // Transpose the elements by generating a load from the
//                      // reverse indices.
//                      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
//                      return builder.create<affine::AffineLoadOp>(loc, input,
//                                                                  reverseIvs);
//                    });
//     return success();
//   }
// };
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Transpose operations (支持 transp 参数)
//===----------------------------------------------------------------------===//
struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto transposeOp = cast<riscv::TransposeOp>(op);
    auto transpAttrList = transposeOp.getTransp().getValue();
    SmallVector<int64_t, 4> transpDims; 
    for (mlir::Attribute attr : transpAttrList) {
      auto intAttr = dyn_cast<mlir::IntegerAttr>(attr);
      transpDims.push_back(intAttr.getValue().getSExtValue());
    }
    lowerOpToLoops(op, operands, rewriter,
                   [loc, transpDims](OpBuilder &builder, ValueRange memRefOperands,
                                     ValueRange loopIvs) {
          
                     riscv::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value inputMemRef = transposeAdaptor.getInput();

                     SmallVector<Value, 4> transposedIvs;
                     for (int64_t dimIdx : transpDims) {
                       transposedIvs.push_back(loopIvs[dimIdx]);
                     }
                     return builder.create<affine::AffineLoadOp>(
                         loc, inputMemRef, transposedIvs);
                   });

    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Reduce operations
//===----------------------------------------------------------------------===//
//******注：此处注释掉的代码是保留规约维度的代码，满足案例：%res1 = "riscv.reduce"(%0) {kind = "sum",dim=[0]} : (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>******//
// struct ReduceOpLowering : public ConversionPattern {
//   ReduceOpLowering(MLIRContext *ctx)
//       : ConversionPattern(riscv::ReduceOp::getOperationName(), 1, ctx) {}

//   mlir::LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto loc = op->getLoc();
//     auto reduceOp = cast<riscv::ReduceOp>(op);

//     auto inputTensorType = dyn_cast<RankedTensorType>(reduceOp.getInput().getType());
//     auto resultTensorType = dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
 
//     std::string reduceKind = reduceOp.getKind().str();
//     ArrayAttr dimAttr = reduceOp.getDimAttr();

//     SmallVector<int64_t, 4> reduceDims;
//     for (auto dimElem : dimAttr.getValue()) {
//       auto dimInt = dyn_cast<IntegerAttr>(dimElem);
 
//       reduceDims.push_back(dimInt.getInt());
//     }

//     auto resultMemRefType = convertTensorToMemRef(resultTensorType);
//     Value resultAlloc = insertAllocAndDealloc(resultMemRefType, loc, rewriter);

//     auto elemType = inputTensorType.getElementType();
//     Value initVal;
//     if (isa<FloatType>(elemType)) {
//       auto floatType = cast<FloatType>(elemType);
//             if (reduceKind == "sum" || reduceKind == "add") {
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
//       } else if (reduceKind == "mul") {
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, 1.0));
//       } else if (reduceKind == "max") {
      
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, -INFINITY));
//       } else if (reduceKind == "min") {
        
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, INFINITY));
//       }
//     } else if (isa<IntegerType>(elemType)) {
//       auto intType = cast<IntegerType>(elemType);
//       bool isSigned = intType.isSigned();
//       unsigned bitWidth = intType.getWidth(); 
//       if (reduceKind == "sum" || reduceKind == "add") {
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
//       } else if (reduceKind == "mul") {
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, 1));
//       } else if (reduceKind == "max") {
//         int64_t minVal;
//         if (isSigned) {
//           minVal = -(1LL << (bitWidth - 1));
//         } else {
//           minVal = 0;
//         }
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, minVal));
//       } else if (reduceKind == "min") {
//         int64_t maxVal;
//         if (isSigned) {
//           maxVal = (1LL << (bitWidth - 1)) - 1;
//         } else {
//           maxVal = (1ULL << bitWidth) - 1;
//         }
//         initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, maxVal));
//       } 
//     } 

//     ArrayRef<int64_t> inputShapeRef = inputTensorType.getShape();
//     SmallVector<int64_t, 4> inputShape(inputShapeRef.begin(), inputShapeRef.end());
//     ArrayRef<int64_t> resultShapeRef = resultTensorType.getShape();
//     SmallVector<int64_t, 4> resultShape(resultShapeRef.begin(), resultShapeRef.end());

//     SmallVector<int64_t, 4> keepDims;
//     SmallVector<int64_t, 4> keepLowerBounds, keepUpperBounds, keepSteps;
//     for (size_t dimIdx = 0; dimIdx < inputShape.size(); dimIdx++) {
//       bool isReduceDim = llvm::is_contained(reduceDims, (int64_t)dimIdx);
//       if (!isReduceDim) {
//         keepDims.push_back(dimIdx);
//         keepLowerBounds.push_back(0);
//         keepUpperBounds.push_back(inputShape[dimIdx]);
//         keepSteps.push_back(1);
//       }
//     }

//     affine::buildAffineLoopNest(
//         rewriter, loc, keepLowerBounds, keepUpperBounds, keepSteps,
//         [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange keepIVs) {

//           SmallVector<Value, 4> outputIndices;
//           int keepIVIdx = 0;
//           for (size_t dimIdx = 0; dimIdx < resultTensorType.getRank(); dimIdx++) {
//             if (llvm::is_contained(reduceDims, (int64_t)dimIdx)) {
//               outputIndices.push_back(nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0));
//             } else {
            
//               outputIndices.push_back(keepIVs[keepIVIdx++]);
//             }
//           }

//           nestedBuilder.create<affine::AffineStoreOp>(nestedLoc, initVal, resultAlloc, outputIndices);

//           SmallVector<int64_t, 4> reduceLowerBounds, reduceUpperBounds, reduceSteps;
//           for (int64_t reduceDim : reduceDims) {
//             reduceLowerBounds.push_back(0);
//             reduceUpperBounds.push_back(inputShape[reduceDim]);
//             reduceSteps.push_back(1);
//           }

//           affine::buildAffineLoopNest(
//               nestedBuilder, nestedLoc, reduceLowerBounds, reduceUpperBounds, reduceSteps,
//               [&](OpBuilder &innerBuilder, Location innerLoc, ValueRange reduceIVs) {
               
//                 SmallVector<Value, 4> inputIndices;
//                 int reduceIVIdx = 0, keepIVIdxInner = 0;
//                 for (size_t dimIdx = 0; dimIdx < inputTensorType.getRank(); dimIdx++) {
//                   if (llvm::is_contained(reduceDims, (int64_t)dimIdx)) {
//                     inputIndices.push_back(reduceIVs[reduceIVIdx++]); 
//                   } else {
//                     inputIndices.push_back(keepIVs[keepIVIdxInner++]); 
//                   }
//                 }

//                 Value inputVal = innerBuilder.create<affine::AffineLoadOp>(innerLoc, operands[0], inputIndices);

//                 Value currentResult = innerBuilder.create<affine::AffineLoadOp>(innerLoc, resultAlloc, outputIndices);

//                 Value newResult;
//                 if (reduceKind == "sum" || reduceKind == "add") {
//                   newResult = isa<FloatType>(elemType)
//                               ? innerBuilder.create<arith::AddFOp>(innerLoc, currentResult, inputVal).getResult()
//                               : innerBuilder.create<arith::AddIOp>(innerLoc, currentResult, inputVal).getResult();
//                 } else if (reduceKind == "mul") {
//                   newResult = isa<FloatType>(elemType)
//                               ? innerBuilder.create<arith::MulFOp>(innerLoc, currentResult, inputVal).getResult()
//                               : innerBuilder.create<arith::MulIOp>(innerLoc, currentResult, inputVal).getResult();
//                 } else if (reduceKind == "max") {
//                   if (isa<FloatType>(elemType)) {
//                     newResult = innerBuilder.create<arith::MaxNumFOp>(innerLoc, currentResult, inputVal).getResult();
//                   } else if (isa<IntegerType>(elemType)) {
//                     auto intType = cast<IntegerType>(elemType);
//                     if (intType.isSigned()) {
//                       newResult = innerBuilder.create<arith::MaxSIOp>(innerLoc, currentResult, inputVal).getResult();
//                     } else {
//                       newResult = innerBuilder.create<arith::MaxUIOp>(innerLoc, currentResult, inputVal).getResult();
//                     }
//                   }

//                 } else if (reduceKind == "min") {
//                   if (isa<FloatType>(elemType)) {
      
//                     newResult = innerBuilder.create<arith::MinNumFOp>(innerLoc, currentResult, inputVal).getResult();
//                   } else if (isa<IntegerType>(elemType)) {
//                     auto intType = cast<IntegerType>(elemType);
//                     if (intType.isSigned()) {
//                       newResult = innerBuilder.create<arith::MinSIOp>(innerLoc, currentResult, inputVal).getResult();
//                     } else {
//                       newResult = innerBuilder.create<arith::MinUIOp>(innerLoc, currentResult, inputVal).getResult();
//                     }
//                   }
//                 } 
//                 innerBuilder.create<affine::AffineStoreOp>(innerLoc, newResult, resultAlloc, outputIndices);
//               });
//         });

//     rewriter.replaceOp(op, resultAlloc);
//     return mlir::success();
//   }
// };
struct ReduceOpLowering : public ConversionPattern {
  ReduceOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::ReduceOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto reduceOp = cast<riscv::ReduceOp>(op);

    auto inputTensorType = dyn_cast<RankedTensorType>(reduceOp.getInput().getType());
    auto resultTensorType = dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
 
    std::string reduceKind = reduceOp.getKind().str();
    ArrayAttr dimAttr = reduceOp.getDimAttr();

    SmallVector<int64_t, 4> reduceDims;
    for (auto dimElem : dimAttr.getValue()) {
      auto dimInt = dyn_cast<IntegerAttr>(dimElem);
      reduceDims.push_back(dimInt.getInt());
    }

    auto resultMemRefType = convertTensorToMemRef(resultTensorType);
    Value resultAlloc = insertAllocAndDealloc(resultMemRefType, loc, rewriter);

    auto elemType = inputTensorType.getElementType();
    Value initVal;
    if (isa<FloatType>(elemType)) {
      auto floatType = cast<FloatType>(elemType);
      if (reduceKind == "sum" || reduceKind == "add") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
      } else if (reduceKind == "mul") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, 1.0));
      } else if (reduceKind == "max") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, -INFINITY));
      } else if (reduceKind == "min") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, INFINITY));
      }
    } else if (isa<IntegerType>(elemType)) {
      auto intType = cast<IntegerType>(elemType);
      bool isSigned = intType.isSigned();
      unsigned bitWidth = intType.getWidth(); 
      if (reduceKind == "sum" || reduceKind == "add") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
      } else if (reduceKind == "mul") {
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, 1));
      } else if (reduceKind == "max") {
        int64_t minVal;
        if (isSigned) {
          minVal = -(1LL << (bitWidth - 1));
        } else {
          minVal = 0;
        }
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, minVal));
      } else if (reduceKind == "min") {
        int64_t maxVal;
        if (isSigned) {
          maxVal = (1LL << (bitWidth - 1)) - 1;
        } else {
          maxVal = (1ULL << bitWidth) - 1;
        }
        initVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, maxVal));
      } 
    } 

    ArrayRef<int64_t> inputShapeRef = inputTensorType.getShape();
    SmallVector<int64_t, 4> inputShape(inputShapeRef.begin(), inputShapeRef.end());
    ArrayRef<int64_t> resultShapeRef = resultTensorType.getShape();
    SmallVector<int64_t, 4> resultShape(resultShapeRef.begin(), resultShapeRef.end());

    SmallVector<int64_t, 4> keepDims;
    SmallVector<int64_t, 4> keepLowerBounds, keepUpperBounds, keepSteps;
    for (size_t dimIdx = 0; dimIdx < inputShape.size(); dimIdx++) {
      bool isReduceDim = llvm::is_contained(reduceDims, (int64_t)dimIdx);
      if (!isReduceDim) {
        keepDims.push_back(dimIdx);
        keepLowerBounds.push_back(0);
        keepUpperBounds.push_back(inputShape[dimIdx]);
        keepSteps.push_back(1);
      }
    }

    affine::buildAffineLoopNest(
        rewriter, loc, keepLowerBounds, keepUpperBounds, keepSteps,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange keepIVs) {

          SmallVector<Value, 4> outputIndices(keepIVs.begin(), keepIVs.end());

          nestedBuilder.create<affine::AffineStoreOp>(nestedLoc, initVal, resultAlloc, outputIndices);

          SmallVector<int64_t, 4> reduceLowerBounds, reduceUpperBounds, reduceSteps;
          for (int64_t reduceDim : reduceDims) {
            reduceLowerBounds.push_back(0);
            reduceUpperBounds.push_back(inputShape[reduceDim]);
            reduceSteps.push_back(1);
          }

          affine::buildAffineLoopNest(
              nestedBuilder, nestedLoc, reduceLowerBounds, reduceUpperBounds, reduceSteps,
              [&](OpBuilder &innerBuilder, Location innerLoc, ValueRange reduceIVs) {
               
                SmallVector<Value, 4> inputIndices;
                int reduceIVIdx = 0, keepIVIdxInner = 0;
                for (size_t dimIdx = 0; dimIdx < inputTensorType.getRank(); dimIdx++) {
                  if (llvm::is_contained(reduceDims, (int64_t)dimIdx)) {
                    inputIndices.push_back(reduceIVs[reduceIVIdx++]); 
                  } else {
                    inputIndices.push_back(keepIVs[keepIVIdxInner++]); 
                  }
                }

                Value inputVal = innerBuilder.create<affine::AffineLoadOp>(innerLoc, operands[0], inputIndices);

                Value currentResult = innerBuilder.create<affine::AffineLoadOp>(innerLoc, resultAlloc, outputIndices);

                Value newResult;
                if (reduceKind == "sum" || reduceKind == "add") {
                  newResult = isa<FloatType>(elemType)
                              ? innerBuilder.create<arith::AddFOp>(innerLoc, currentResult, inputVal).getResult()
                              : innerBuilder.create<arith::AddIOp>(innerLoc, currentResult, inputVal).getResult();
                } else if (reduceKind == "mul") {
                  newResult = isa<FloatType>(elemType)
                              ? innerBuilder.create<arith::MulFOp>(innerLoc, currentResult, inputVal).getResult()
                              : innerBuilder.create<arith::MulIOp>(innerLoc, currentResult, inputVal).getResult();
                } else if (reduceKind == "max") {
                  if (isa<FloatType>(elemType)) {
                    newResult = innerBuilder.create<arith::MaxNumFOp>(innerLoc, currentResult, inputVal).getResult();
                  } else if (isa<IntegerType>(elemType)) {
                    auto intType = cast<IntegerType>(elemType);
                    if (intType.isSigned()) {
                      newResult = innerBuilder.create<arith::MaxSIOp>(innerLoc, currentResult, inputVal).getResult();
                    } else {
                      newResult = innerBuilder.create<arith::MaxUIOp>(innerLoc, currentResult, inputVal).getResult();
                    }
                  }

                } else if (reduceKind == "min") {
                  if (isa<FloatType>(elemType)) {
                    newResult = innerBuilder.create<arith::MinNumFOp>(innerLoc, currentResult, inputVal).getResult();
                  } else if (isa<IntegerType>(elemType)) {
                    auto intType = cast<IntegerType>(elemType);
                    if (intType.isSigned()) {
                      newResult = innerBuilder.create<arith::MinSIOp>(innerLoc, currentResult, inputVal).getResult();
                    } else {
                      newResult = innerBuilder.create<arith::MinUIOp>(innerLoc, currentResult, inputVal).getResult();
                    }
                  }
                } 
                innerBuilder.create<affine::AffineStoreOp>(innerLoc, newResult, resultAlloc, outputIndices);
              });
        });

    rewriter.replaceOp(op, resultAlloc);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToArith ConversionPattern: LoweredBinaryOp 
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Type Promote function define...........................BEGIN
//===----------------------------------------------------------------------===//

// Define priority for type kinds (higher value means higher priority)
static int getKindPriority(Type type) {
  // if (type.isa<FloatType>()) return 2;   // Floating-point types have higher priority than integers
  if (isa<FloatType>(type)) return 2;
  if (isa<IntegerType>(type) || isa<IndexType>(type)) return 1;
  // if (type.isa<IntegerType>() || type.isa<IndexType>()) return 1;   // Integer types
  return 0;   // Other types
}

// Get type width (return element type width for composite types)
static unsigned getTypeWidth(Type type) {
  Type elemType = type;
  // if (auto vecType = type.dyn_cast<VectorType>())
  //   elemType = vecType.getElementType();
  // else if (auto tensorType = type.dyn_cast<TensorType>())
  //   elemType = tensorType.getElementType();
  // else if (auto memRefType = type.dyn_cast<MemRefType>())
  //   elemType = memRefType.getElementType();

  // if (auto intType = elemType.dyn_cast<IntegerType>())
  //   return intType.getWidth();
  // if (auto floatType = elemType.dyn_cast<FloatType>())
  //   return floatType.getWidth();
  if (auto vecType = dyn_cast<VectorType>(type))
    elemType = vecType.getElementType();
  else if (auto tensorType = dyn_cast<TensorType>(type))
    elemType = tensorType.getElementType();
  else if (auto memRefType = dyn_cast<MemRefType>(type))
    elemType = memRefType.getElementType();

  if (auto intType = dyn_cast<IntegerType>(elemType))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(elemType))
    return floatType.getWidth();
  // if (elemType.isa<IndexType>())
  if (isa<IndexType>(elemType))
    return 64;  // 索引类型为64位？
  return 0;
}
// Check if the type is bfloat16
static bool isBFloat16(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
  // if (auto floatType = type.dyn_cast<FloatType>()) {
    return floatType.isBF16();
  }
  return false;
}

// Check if the type is fp8
static bool isFP8(Type type) {
    if (auto floatType = dyn_cast<FloatType>(type)){
    // if (auto floatType = type.dyn_cast<FloatType>()){
    // 判断位宽是否为 8
    return floatType.getWidth() == 8;
  }
  return false;
}
// Get element type
static Type getElementType(Type type) {
  // if (auto vecType = type.dyn_cast<VectorType>())
  //   return vecType.getElementType();
  // else if (auto tensorType = type.dyn_cast<TensorType>())
  //   return tensorType.getElementType();
  // else if (auto memRefType = type.dyn_cast<MemRefType>())
  //   return memRefType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  else if (auto tensorType = dyn_cast<TensorType>(type))
    return tensorType.getElementType();
  else if (auto memRefType = dyn_cast<MemRefType>(type))
    return memRefType.getElementType();
  return type;
}

// Create a new type with the same structure as the original type but different element type
static Type getNewTypeWithElementType(Type originalType, Type newElementType) {
  // if (auto vecType = originalType.dyn_cast<VectorType>())
  //   return VectorType::get(vecType.getShape(), newElementType);
  // else if (auto tensorType = originalType.dyn_cast<TensorType>()) {
  //   if (auto rankedType = tensorType.dyn_cast<RankedTensorType>())
  //     return RankedTensorType::get(rankedType.getShape(), newElementType);
  //   return UnrankedTensorType::get(newElementType);
  // } else if (auto memRefType = originalType.dyn_cast<MemRefType>())
  //   return MemRefType::get(memRefType.getShape(), newElementType, 
  //                         memRefType.getLayout(), memRefType.getMemorySpace());
  if (auto vecType = dyn_cast<VectorType>(originalType))
    return VectorType::get(vecType.getShape(), newElementType);
  else if (auto tensorType = dyn_cast<TensorType>(originalType)) {
    if (auto rankedType = dyn_cast<RankedTensorType>(tensorType))
      return RankedTensorType::get(rankedType.getShape(), newElementType);
    return UnrankedTensorType::get(newElementType);
  } else if (auto memRefType = dyn_cast<MemRefType>(originalType))
    return MemRefType::get(memRefType.getShape(), newElementType, 
                          memRefType.getLayout(), memRefType.getMemorySpace());                         
  return newElementType;  // Scalar type
}

// Determine the target type according to the type promotion rules
static Type getPromotedType(Type type1, Type type2, MLIRContext *ctx) {  
  Type elemType1 = getElementType(type1);
  Type elemType2 = getElementType(type2);
    // llvm::outs() << "**********elemType1: " << elemType1 << "\n"; 

    // llvm::outs() << "**********elemType2: " << elemType2 << "\n"; 

  // Rule1. Kind. If one tensor is of a dtype of a higher kind,
  // the other tensor is promoted to this dtype: (int32, bfloat16) -> bfloat16
  int kind1 = getKindPriority(elemType1);
  int kind2 = getKindPriority(elemType2);
  if (kind1 != kind2) {
    Type higherKindType = kind1 > kind2 ? elemType1 : elemType2;
    return getNewTypeWithElementType(type1, higherKindType);
    // llvm::outs() << "**********higherKindType: " << higherKindType << "\n"; 
      
  }

  // Rule2. Width If both tensors are of dtypes of the same kind, and one of them is of a higher width,
  // the other one is promoted to this dtype: (float32, float16) -> float32
  unsigned width1 = getTypeWidth(elemType1);
  unsigned width2 = getTypeWidth(elemType2);
  if ( width1 != width2) {
    Type widerType = width1 > width2 ? elemType1 : elemType2;
    return getNewTypeWithElementType(type1, widerType);
  }

  //Rule3. Prefer float16 If both tensors are of the same width and signedness but different dtypes (float16 and bfloat16 or different fp8 types),
  // they are both promoted to float16. (float16, bfloat16) -> float16
  if (isa<FloatType>(elemType1) && isa<FloatType>(elemType2)) {
  // if (elemType1.isa<FloatType>() && elemType2.isa<FloatType>()) {

      //     llvm::outs() << "**********FloatType::getF16: " << isBFloat16(elemType1) << "\n"; 
      // llvm::outs() << "**********FloatType::getF16: " << isBFloat16(elemType2) << "\n"; 
    if ((isBFloat16(elemType1) && (!isBFloat16(elemType2))) ||
        (!isBFloat16(elemType1) && isBFloat16(elemType2)) ||
        (isFP8(elemType1) && !isFP8(elemType2)) ||
        (!isFP8(elemType1) && isFP8(elemType2))) {
          mlir::Builder builder(ctx);
          Type float16 = builder.getF16Type();
          // Type float16 = Builder::getF16Type(ctx);
      // llvm::outs() << "**********FloatType::getF16: " << getNewTypeWithElementType(type1, float16) << "\n"; 
      return getNewTypeWithElementType(type1, float16);
    }
  }

  //Rule4. Prefer unsigned Otherwise (same width, different signedness), 
  //they are promoted to the unsigned dtype: (int32, uint32) -> uint32
  // if (auto intType1 = elemType1.dyn_cast<IntegerType>()) {
  //   if (auto intType2 = elemType2.dyn_cast<IntegerType>()) {
    if (auto intType1 = dyn_cast<IntegerType>(elemType1)) {
    if (auto intType2 = dyn_cast<IntegerType>(elemType2)) {
      if (intType1.isSigned() && intType2.isUnsigned())
        return getNewTypeWithElementType(type1, intType2);
      if (intType1.isUnsigned() && intType2.isSigned())
        return getNewTypeWithElementType(type1, intType1);
    }
  }
  // If none of the rules apply, retain the original type.
  return type1;
}
  // 类型转换函数：将值转换为目标类型
  static Value castToType(OpBuilder &builder, Location loc, Value value, Type targetType) {
    Type sourceType = value.getType(); 

    if (sourceType == targetType) return value;

    Type sourceElemType = getElementType(sourceType);
    Type targetElemType = getElementType(targetType);

    // 对于复合类型，先转换元素类型
    if (auto vecType = dyn_cast<VectorType>(sourceType)) {
    // if (auto vecType = sourceType.dyn_cast<VectorType>()) {

    // .........................vector类型转换 .........................
        // if (sourceElemType.isa<IntegerType>() && targetElemType.isa<IntegerType>()) {
        //   return builder.create<arith::IndexCastOp>(loc, targetType, value);
        // }
        // if (sourceElemType.isa<IntegerType>() && targetElemType.isa<FloatType>()) {
        //   return builder.create<arith::SIToFPOp>(loc, targetType, value);
        // }
        // if (sourceElemType.isa<FloatType>() && targetElemType.isa<IntegerType>()) {
        //   return builder.create<arith::FPToSIOp>(loc, targetType, value);
        // }
        if (isa<IntegerType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
          return builder.create<arith::IndexCastOp>(loc, targetType, value);
        }
        if (isa<IntegerType>(sourceElemType) && isa<FloatType>(targetElemType)) {
          return builder.create<arith::SIToFPOp>(loc, targetType, value);
        }
        if (isa<FloatType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
          return builder.create<arith::FPToSIOp>(loc, targetType, value);
        }
        // if (auto srcF = sourceElemType.dyn_cast<FloatType>()) {
        //       if (auto dstF = targetElemType.dyn_cast<FloatType>()) {
        if (auto srcF = dyn_cast<FloatType>(sourceElemType)) {
              if (auto dstF = dyn_cast<FloatType>(targetElemType)) {
                unsigned srcW = srcF.getWidth();
                unsigned dstW = dstF.getWidth();
                if (srcW < dstW) {
                  // f16->f32, bf16->f32, f32->f64
                  return builder.create<arith::ExtFOp>(loc, targetType, value);
                } else if (srcW == dstW) {
                  // 同宽但格式不同，bf16->f16
                return builder.create<arith::BitcastOp>(loc, targetType, value);
                } 
              }
      }

    } else if (auto memRefType = dyn_cast<mlir::MemRefType>(sourceType)) {
    // } else if (auto memRefType = sourceType.dyn_cast<mlir::MemRefType>()) {

      //  auto valueMemRefType = value.getType().cast<MemRefType>();
      // llvm::outs() << "**********memRefType*****: " << memRefType << "\n";  
      auto sourceElemType = memRefType.getElementType();
      // auto targetMemRefType = targetType.cast<MemRefType>();  
      auto targetMemRefType = cast<MemRefType>(targetType);  

      auto targetElemType = targetMemRefType.getElementType();
 
      auto resultAlloc = insertAllocAndDealloc_OpBuilder(targetMemRefType, loc, builder);

      int64_t rank = memRefType.getRank();
      SmallVector<int64_t, 4> lowerBounds(rank, 0);
      SmallVector<int64_t, 4> steps(rank, 1);
      SmallVector<int64_t, 4> upperBounds(memRefType.getShape().begin(), memRefType.getShape().end());

      affine::buildAffineLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            auto loadedElem = nestedBuilder.create<affine::AffineLoadOp>(loc, value, ivs);  // value已是MemRef，直接加载
            Value convertedElem;
            // if (sourceElemType.isa<IntegerType>() && targetElemType.isa<IntegerType>()) {
            //   convertedElem = nestedBuilder.create<arith::IndexCastOp>(loc, targetElemType, loadedElem);
            // } else if (sourceElemType.isa<IntegerType>() && targetElemType.isa<FloatType>()) {
            //   convertedElem = nestedBuilder.create<arith::SIToFPOp>(loc, targetElemType, loadedElem);
            // } else if (sourceElemType.isa<FloatType>() && targetElemType.isa<IntegerType>()) {
            //   convertedElem = nestedBuilder.create<arith::FPToSIOp>(loc, targetElemType, loadedElem);
            // } else if (auto srcF = sourceElemType.dyn_cast<FloatType>()) {
            if (isa<IntegerType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
              convertedElem = nestedBuilder.create<arith::IndexCastOp>(loc, targetElemType, loadedElem);
            } else if (isa<IntegerType>(sourceElemType) && isa<FloatType>(targetElemType)) {
              convertedElem = nestedBuilder.create<arith::SIToFPOp>(loc, targetElemType, loadedElem);
            } else if (isa<FloatType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
              convertedElem = nestedBuilder.create<arith::FPToSIOp>(loc, targetElemType, loadedElem);
            // } else if (auto srcF = sourceElemType.dyn_cast<FloatType>()) {
            //   if (auto dstF = targetElemType.dyn_cast<FloatType>()) {
            } else if (auto srcF = dyn_cast<FloatType>(sourceElemType)) {
              if (auto dstF = dyn_cast<FloatType>(targetElemType)) {
                unsigned srcW = srcF.getWidth();
                unsigned dstW = dstF.getWidth();
                   if (srcW < dstW) {
                  convertedElem = nestedBuilder.create<arith::ExtFOp>(loc, targetElemType, loadedElem.getResult());
                } else if (srcW == dstW) {
                  convertedElem = nestedBuilder.create<arith::BitcastOp>(loc, targetElemType, loadedElem);
                } 
              }
            } 
            nestedBuilder.create<affine::AffineStoreOp>(loc, convertedElem, resultAlloc, ivs);
          });
      return resultAlloc;
    }

    // .........................标量类型转换 .........................
    if (isa<IntegerType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
    // if (sourceElemType.isa<IntegerType>() && targetElemType.isa<IntegerType>()) {
      return builder.create<arith::IndexCastOp>(loc, targetType, value);
    }
    if (isa<IntegerType>(sourceElemType) && isa<FloatType>(targetElemType)) {
    // if (sourceElemType.isa<IntegerType>() && targetElemType.isa<FloatType>()) {
      return builder.create<arith::SIToFPOp>(loc, targetType, value);
    }
    if (isa<FloatType>(sourceElemType) && isa<IntegerType>(targetElemType)) {
    // if (sourceElemType.isa<FloatType>() && targetElemType.isa<IntegerType>()) {
      return builder.create<arith::FPToSIOp>(loc, targetType, value);
    }
    // if (auto srcF = sourceElemType.dyn_cast<FloatType>()) {
    //   if (auto dstF = targetElemType.dyn_cast<FloatType>()) {
    if (auto srcF = dyn_cast<FloatType>(sourceElemType)) {
      if (auto dstF = dyn_cast<FloatType>(targetElemType)) {
        unsigned srcW = srcF.getWidth();
        unsigned dstW = dstF.getWidth();
        if (srcW < dstW) {
          // f16->f32, bf16->f32, f32->f64
          return builder.create<arith::ExtFOp>(loc, targetType, value);
        } else if (srcW == dstW) {
          // 同宽但格式不同,bf16->f16
          return builder.create<arith::BitcastOp>(loc, targetType, value);
        } 
      }
    }

    return value;
  }
//===----------------------------------------------------------------------===//
// Type Promote function define...........................END
//===----------------------------------------------------------------------===//
static bool isSignlessIntegerLike(Type type) {
  //无符号整数或索引
  if (auto intType = dyn_cast<IntegerType>(type)) {
  // if (auto intType = type.dyn_cast<IntegerType>()) {
    return intType.isSignless(); 
  }
  if (isa<IndexType>(type)) {
  // if (type.isa<IndexType>()) {
    return true; 
  }

  //向量
  // if (auto vecType = type.dyn_cast<VectorType>()) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return isSignlessIntegerLike(vecType.getElementType());
  }

  //张量
  // if (auto tensorType = type.dyn_cast<TensorType>()) {
  if (auto tensorType = dyn_cast<TensorType>(type)) {  
    return isSignlessIntegerLike(tensorType.getElementType());
  }

  //其他类型（如浮点、memref）均不匹配
  return false;
}

template <typename BinaryOp>
struct BinaryOpArithMap {};

template <>
struct BinaryOpArithMap<riscv::AddOp> {
  using IntArithOp = arith::AddIOp;   
  using FloatArithOp = arith::AddFOp; 
};

template <>
struct BinaryOpArithMap<riscv::SubOp> {
  using IntArithOp = arith::SubIOp;   
  using FloatArithOp = arith::SubFOp; 
};

template <>
struct BinaryOpArithMap<riscv::MulOp> {
  using IntArithOp = arith::MulIOp;  
  using FloatArithOp = arith::MulFOp; 
};
template <>
struct BinaryOpArithMap<riscv::OrIOp> {
  using IntArithOp = arith::OrIOp; 
};
template <>
struct BinaryOpArithMap<riscv::XOrIOp> {
  using IntArithOp = arith::XOrIOp; 
 
};
template <>
struct BinaryOpArithMap<riscv::AndIOp> {
  using IntArithOp = arith::AndIOp; 
 
};

template <>
struct BinaryOpArithMap<riscv::DivOp> {
  using FloatArithOp = arith::DivFOp; 
  using SignedIntArithOp = arith::DivSIOp;   // 有符号整数除法
  using UnsignedIntArithOp = arith::DivUIOp; // 无符号整数除法
};

template <>
struct BinaryOpArithMap<riscv::MaxOp> {
  // using FloatArithOp = arith::DivFOp; 
  using SignedIntArithOp = arith::MaxSIOp;   
  using UnsignedIntArithOp = arith::MaxUIOp; 
};
template <>
struct BinaryOpArithMap<riscv::MinOp> {
  // using FloatArithOp = arith::DivFOp; 
  using SignedIntArithOp = arith::MinSIOp;   
  using UnsignedIntArithOp = arith::MinUIOp; 
};

// -------------------------- 检测操作类型是否存在 --------------------------
// 检测是否存在IntArithOp
template <typename T, typename = void>
struct HasIntArithOp : std::false_type {};
template <typename T>
struct HasIntArithOp<T, std::void_t<typename T::SignedIntArithOp>> : std::true_type {};

// 检测是否存在FloatArithOp
template <typename T, typename = void>
struct HasFloatArithOp : std::false_type {};
template <typename T>
struct HasFloatArithOp<T, std::void_t<typename T::UnsignedIntArithOp>> : std::true_type {};

template <typename ArithMap, typename = void>
struct GetIntOpType {
  using type = void; 
};
template <typename ArithMap>
struct GetIntOpType<ArithMap, std::void_t<typename ArithMap::IntArithOp>> {
  using type = typename ArithMap::IntArithOp; 
};

// 获取FloatArithOp
template <typename ArithMap, typename = void>
struct GetFloatOpType {
  using type = void; 
};
template <typename ArithMap>
struct GetFloatOpType<ArithMap, std::void_t<typename ArithMap::FloatArithOp>> {
  using type = typename ArithMap::FloatArithOp; 
};

// -------------------- 支持 DivSIOp/DivUIOp --------------------

template <typename ArithMap, typename = void>
struct GetSignedIntOpType {
  using type = void;
};
template <typename ArithMap>
struct GetSignedIntOpType<ArithMap, std::void_t<typename ArithMap::SignedIntArithOp>> {
  using type = typename ArithMap::SignedIntArithOp;
};

//获取 UnsignedIntArithOp
template <typename ArithMap, typename = void>
struct GetUnsignedIntOpType {
  using type = void;
};
template <typename ArithMap>
struct GetUnsignedIntOpType<ArithMap, std::void_t<typename ArithMap::UnsignedIntArithOp>> {
  using type = typename ArithMap::UnsignedIntArithOp;
};


template <typename IntOp, typename FloatOp,
          typename SignedIntOp = void, typename UnsignedIntOp = void>
  // template <typename IntOp, typename FloatOp>
  static Value createArithOp(OpBuilder &builder, Location loc,
                             Type resultType, Value lhs, Value rhs) {
  // llvm::outs() << "lhs 类型: " << lhs.getType() << "\n";  
  // llvm::outs() << "rhs 类型: " << rhs.getType() << "\n";

  Type elemType = getElementType(resultType);
    
    // 浮点分支：仅当 FloatOp 存在时执行
  if constexpr (!std::is_void_v<FloatOp>) {
    if (isa<FloatType>(elemType)) {
    // if (elemType.isa<FloatType>()) {
      return builder.create<FloatOp>(loc, resultType, lhs, rhs).getResult();
    }
  }
    // 整数分支：仅当 IntOp 存在时执行
  if constexpr (!std::is_void_v<IntOp>) {
    if (isSignlessIntegerLike(elemType)) {
      return builder.create<IntOp>(loc, resultType, lhs, rhs).getResult();
    }
  }
  // 有符号整数
  if constexpr (!std::is_void_v<SignedIntOp>) {
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
    // if (auto intType = elemType.dyn_cast<IntegerType>()) {
      if (intType.isSignless() || intType.isSigned()) {   
        return builder.create<SignedIntOp>(loc, resultType, lhs, rhs).getResult();
      }
    }
  }

  // 无符号整数
  if constexpr (!std::is_void_v<UnsignedIntOp>) {
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
    // if (auto intType = elemType.dyn_cast<IntegerType>()) {

      if (intType.isUnsigned()) {
        return builder.create<UnsignedIntOp>(loc, resultType, lhs, rhs).getResult();
      }
    }
  }
    return nullptr;
}
  

template <typename BinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    typename BinaryOp::Adaptor adaptor(operands);   
    auto binaryOp = cast<BinaryOp>(op);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type lhsType = lhs.getType();
    Type rhsType = rhs.getType();
    // 确定提升后的目标类型
    Type resultType = getPromotedType(lhsType, rhsType, rewriter.getContext());
      // llvm::outs() << "**********getPromotedType*****: " << resultType << "\n";  
    // 对操作数进行类型转换
    Value castLhs = castToType(rewriter, loc, lhs, resultType);
    Value castRhs = castToType(rewriter, loc, rhs, resultType);
    // llvm::outs() << "**********castLhs*****: " << castLhs << "\n";  
    // llvm::outs() << "**********castRhs*****: " << castRhs << "\n";  

    if (!castLhs || !castRhs) {
    return op->emitError("failed to cast operands to promoted type");
  }
    // -------------------------- 标量处理分支 --------------------------
    // bool isScalar = !lhsType.isa<mlir::VectorType>() && !lhsType.isa<mlir::TensorType>() && 
    //                 !lhsType.isa<mlir::MemRefType>();
    bool isScalar = !isa<mlir::VectorType>(lhsType) && !isa<mlir::TensorType>(lhsType) && 
                !isa<mlir::MemRefType>(lhsType);
    if (isScalar) {
      // 动态创建arith标量操作（依赖BinaryOpArithMap映射）
      using ArithMap = BinaryOpArithMap<BinaryOp>;
      using FloatOp = typename GetFloatOpType<ArithMap>::type;
      using IntOp = typename GetIntOpType<ArithMap>::type;
      using SignedIntOp = typename GetSignedIntOpType<ArithMap>::type;      
      using UnsignedIntOp = typename GetUnsignedIntOpType<ArithMap>::type;  

      // Value loweredResult = createArithOp<IntOp, FloatOp, SignedIntOp, UnsignedIntOp>(rewriter, loc, lhsType, lhs, rhs);  
      Value loweredResult = createArithOp<IntOp, FloatOp, SignedIntOp, UnsignedIntOp>(rewriter, loc, resultType, castLhs, castRhs);  
      
      if (!loweredResult) return failure();
      
      rewriter.replaceOp(op, loweredResult);
      return success();
    }

    // -------------------------- 向量处理分支 --------------------------
    bool isVector = isa<VectorType>(lhsType);
    // bool isVector = lhsType.isa<VectorType>();

    if (isVector) {
      auto vecType = cast<VectorType>(lhsType);
      // auto vecType = lhsType.cast<VectorType>();
      // 动态创建arith向量操作
      using ArithMap = BinaryOpArithMap<BinaryOp>;
      using IntOp = typename GetIntOpType<ArithMap>::type;
      using SignedIntOp = typename GetSignedIntOpType<ArithMap>::type;      
      using UnsignedIntOp = typename GetUnsignedIntOpType<ArithMap>::type;  
      using FloatOp = typename GetFloatOpType<ArithMap>::type;
      // 处理向量类型（VectorType）的形状

      // Value loweredResult = createArithOp<IntOp, FloatOp, SignedIntOp, UnsignedIntOp>(rewriter, loc, vecType, lhs, rhs);  
      Value loweredResult = createArithOp<IntOp, FloatOp, SignedIntOp, UnsignedIntOp>(rewriter, loc, resultType, castLhs, castRhs);  
      
      if (!loweredResult) return failure();
      
      rewriter.replaceOp(op, loweredResult);
      return success();
    }

    // -------------------------- Tensor&MemRef处理分支 --------------------------
    // bool isTensorOrMemRef = lhsType.isa<mlir::TensorType>() || lhsType.isa<mlir::MemRefType>();
    bool isTensorOrMemRef = isa<mlir::TensorType>(lhsType) || isa<mlir::MemRefType>(lhsType);

    if (isTensorOrMemRef) {
      auto lowerFn = [&](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
        typename BinaryOp::Adaptor adaptor(memRefOperands);
        auto loadedLhs = builder.create<affine::AffineLoadOp>(loc, adaptor.getLhs(), loopIvs);
        auto loadedRhs = builder.create<affine::AffineLoadOp>(loc, adaptor.getRhs(), loopIvs);


        // 对加载的值进行类型转换
      Type elemResultType = getElementType(resultType);
      // llvm::outs() << "**********elemResultType*****: " << elemResultType << "\n";  

      Value castLoadedLhs = castToType(builder, loc, loadedLhs, elemResultType);
      Value castLoadedRhs = castToType(builder, loc, loadedRhs, elemResultType);

        using ArithMap = BinaryOpArithMap<BinaryOp>;
        using IntOp = typename GetIntOpType<ArithMap>::type;
        // using SignedIntOp = typename GetSignedIntOpType<ArithMap>::type;      
        // using UnsignedIntOp = typename GetUnsignedIntOpType<ArithMap>::type;  
        using FloatOp = typename GetFloatOpType<ArithMap>::type;

        // return createArithOp<IntOp, FloatOp, SignedIntOp, UnsignedIntOp>(builder, loc, loadedLhs.getType(), loadedLhs, loadedRhs); 
        return createArithOp<IntOp, FloatOp>(builder, loc, elemResultType, castLoadedLhs, castLoadedRhs);
      };

      lowerOpToLoops(op, operands, rewriter, lowerFn);
      return success();
    }

    return op->emitError("unsupported operand type: ") << lhsType;
  }
};

using AddOpLowering = BinaryOpLowering<riscv::AddOp>;    
using SubOpLowering = BinaryOpLowering<riscv::SubOp>;    
using MulOpLowering = BinaryOpLowering<riscv::MulOp>;
using DivOpLowering = BinaryOpLowering<riscv::DivOp>;    


using AndIOpLowering = BinaryOpLowering<riscv::AndIOp>;
using XOrIOpLowering = BinaryOpLowering<riscv::XOrIOp>;
using OrIOpLowering = BinaryOpLowering<riscv::OrIOp>;

using MaxOpLowering = BinaryOpLowering<riscv::MaxOp>;
using MinOpLowering = BinaryOpLowering<riscv::MinOp>;
// template <typename BinaryOp, typename LoweredBinaryOp>
// struct BinaryOpLowering : public ConversionPattern {
//   BinaryOpLowering(MLIRContext *ctx)
//       : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto loc = op->getLoc();
//     lowerOpToLoops(op, operands, rewriter,
//                    [loc](OpBuilder &builder, ValueRange memRefOperands,
//                          ValueRange loopIvs) {
//                      // Generate an adaptor for the remapped operands of the
//                      // BinaryOp. This allows for using the nice named accessors
//                      // that are generated by the ODS.
//                      typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

//                      // Generate loads for the element of 'lhs' and 'rhs' at the
//                      // inner loop.
//                      auto loadedLhs = builder.create<affine::AffineLoadOp>(
//                          loc, binaryAdaptor.getLhs(), loopIvs);
//                      auto loadedRhs = builder.create<affine::AffineLoadOp>(
//                          loc, binaryAdaptor.getRhs(), loopIvs);

//                      // Create the binary operation performed on the loaded
//                      // values.
//                      return builder.create<LoweredBinaryOp>(loc, loadedLhs,
//                                                             loadedRhs);
//                    });
//     return success();
//   }
// };
// using AddFOpLowering = BinaryOpLowering<riscv::AddFOp, arith::AddFOp>;
// using SubFOpLowering = BinaryOpLowering<riscv::SubFOp, arith::SubFOp>;
// using MulFOpLowering = BinaryOpLowering<riscv::MulFOp, arith::MulFOp>;
// using DivFOpLowering = BinaryOpLowering<riscv::DivFOp, arith::DivFOp>;

// using AddIOpLowering = BinaryOpLowering<riscv::AddIOp, arith::AddIOp>;
// using SubIOpLowering = BinaryOpLowering<riscv::SubIOp, arith::SubIOp>;
// using MulIOpLowering = BinaryOpLowering<riscv::MulIOp, arith::MulIOp>;
// using DivSIOpLowering = BinaryOpLowering<riscv::DivSIOp, arith::DivSIOp>;
// using DivUIOpLowering = BinaryOpLowering<riscv::DivUIOp, arith::DivUIOp>;

//===----------------------------------------------------------------------===//
// ToArith ConversionPattern: LoweredUnaryOp 
//===----------------------------------------------------------------------===//
template <typename SourceUnaryOp, typename LoweredUnaryOp>
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(SourceUnaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     typename SourceUnaryOp::Adaptor unaryAdaptor(memRefOperands);

                     auto loadedOperand = builder.create<affine::AffineLoadOp>(
                         loc, unaryAdaptor.getOperand(), loopIvs);  

                     return builder.create<LoweredUnaryOp>(
                         loc, loadedOperand);  
                   });

    return success();
  }
};
using NegFOpLowering = UnaryOpLowering<riscv::NegFOp, arith::NegFOp>;

//===----------------------------------------------------------------------===//
// CmpIOpLowering
//===----------------------------------------------------------------------===//
struct CmpIOpLowering : public ConversionPattern {
  CmpIOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::CmpIOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    riscv::CmpIOpAdaptor adaptor(operands);
    auto cmpOp = cast<riscv::CmpIOp>(op);

    // Get predicate attribute and map to arith comparison predicate
    auto predAttr = cmpOp.getPredicateAttr();
    auto pred = llvm::StringSwitch<arith::CmpIPredicate>(predAttr.getValue())
                  .Case("eq", arith::CmpIPredicate::eq)
                  .Case("ne", arith::CmpIPredicate::ne)
                  .Case("slt", arith::CmpIPredicate::slt)
                  .Case("sle", arith::CmpIPredicate::sle)
                  .Case("sgt", arith::CmpIPredicate::sgt)
                  .Case("sge", arith::CmpIPredicate::sge)
                  .Case("ult", arith::CmpIPredicate::ult)
                  .Case("ule", arith::CmpIPredicate::ule)
                  .Case("ugt", arith::CmpIPredicate::ugt)
                  .Case("uge", arith::CmpIPredicate::uge)
                  .Default([&]() {
                    rewriter.notifyMatchFailure(op, "unsupported predicate: " + predAttr.getValue());
                    return arith::CmpIPredicate::eq;
                  }());

    // Get operands and result type
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    
    // Check if operands are of memref type
    // bool isLhsMemRef = lhs.getType().isa<MemRefType>();
    // bool isRhsMemRef = rhs.getType().isa<MemRefType>();
    bool isLhsMemRef = mlir::isa<mlir::MemRefType>(lhs.getType());
    bool isRhsMemRef = mlir::isa<mlir::MemRefType>(rhs.getType());

    
    // Handle scalar comparison case
    if (!isLhsMemRef && !isRhsMemRef) {
      // Directly create arith.cmpi operation
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, rewriter.getI1Type(), pred, lhs, rhs);
      return success();
    }

    // Handle memref type comparison case
    // auto lhsMemRefType = lhs.getType().cast<MemRefType>();
    // auto rhsMemRefType = rhs.getType().cast<MemRefType>();
    auto lhsMemRefType = cast<MemRefType>(lhs.getType());
    auto rhsMemRefType = cast<MemRefType>(rhs.getType());

    // Create memref for comparison result (element type is i1)
    SmallVector<int64_t, 4> resultShape(lhsMemRefType.getShape().begin(),
                                        lhsMemRefType.getShape().end());
    auto resultMemRefType = MemRefType::get(resultShape, rewriter.getI1Type());
    auto resultAlloc = insertAllocAndDealloc(resultMemRefType, loc, rewriter);

    // Create nested loops to compare each element
    int64_t rank = lhsMemRefType.getRank();
    SmallVector<int64_t, 4> lowerBounds(rank, 0);
    SmallVector<int64_t, 4> steps(rank, 1);
    
    // Build nested loops in lowerOpToLoops style
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, lhsMemRefType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Load current elements from lhs and rhs
          auto loadedLhs = nestedBuilder.create<affine::AffineLoadOp>(loc, lhs, ivs);
          auto loadedRhs = nestedBuilder.create<affine::AffineLoadOp>(loc, rhs, ivs);
          
          // Perform element-wise comparison
          auto cmpResult = nestedBuilder.create<arith::CmpIOp>(
              loc, rewriter.getI1Type(), pred, loadedLhs, loadedRhs);
          
          // Store comparison result
          nestedBuilder.create<affine::AffineStoreOp>(loc, cmpResult, resultAlloc, ivs);
        });

    // Replace original operation with result memref
    rewriter.replaceOp(op, resultAlloc);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// CmpFOpLowering
//===----------------------------------------------------------------------===//

struct CmpFOpLowering : public ConversionPattern {
  CmpFOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::CmpFOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    riscv::CmpFOpAdaptor adaptor(operands);
    auto cmpOp = cast<riscv::CmpFOp>(op);

    // Get predicate attribute and map to arith comparison predicate
    auto predAttr = cmpOp.getPredicateAttr();
    auto pred = llvm::StringSwitch<arith::CmpFPredicate>(predAttr.getValue())
                  .Case("false",  arith::CmpFPredicate::AlwaysFalse)
                  .Case("oeq",    arith::CmpFPredicate::OEQ)
                  .Case("ogt",    arith::CmpFPredicate::OGT)
                  .Case("oge",    arith::CmpFPredicate::OGE)
                  .Case("olt",    arith::CmpFPredicate::OLT)
                  .Case("ole",    arith::CmpFPredicate::OLE)
                  .Case("one",    arith::CmpFPredicate::ONE)
                  .Case("ord",    arith::CmpFPredicate::ORD)
                  .Case("ueq",    arith::CmpFPredicate::UEQ)
                  .Case("ugt",    arith::CmpFPredicate::UGT)
                  .Case("uge",    arith::CmpFPredicate::UGE)
                  .Case("ult",    arith::CmpFPredicate::ULT)
                  .Case("ule",    arith::CmpFPredicate::ULE)
                  .Case("une",    arith::CmpFPredicate::UNE)
                  .Case("uno",    arith::CmpFPredicate::UNO)
                  .Case("true",   arith::CmpFPredicate::AlwaysTrue)
                  .Default([&]() {
                    rewriter.notifyMatchFailure(op, "unsupported predicate: " + predAttr.getValue());
                    return arith::CmpFPredicate::AlwaysFalse;
                  }());

    // Get operands and check types
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
     
    // Check if operands are of memref type
    // bool isLhsMemRef = lhs.getType().isa<MemRefType>();
    // bool isRhsMemRef = rhs.getType().isa<MemRefType>();
    bool isLhsMemRef = mlir::isa<mlir::MemRefType>(lhs.getType());
    bool isRhsMemRef = mlir::isa<mlir::MemRefType>(rhs.getType());
    
    // Handle scalar comparison case
    if (!isLhsMemRef && !isRhsMemRef) {
      // Directly create arith.cmpf operation
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(
          op, rewriter.getI1Type(), pred, lhs, rhs);
      return success();
    }

    // Handle memref type comparison case
    // auto lhsMemRefType = lhs.getType().cast<MemRefType>();
    // auto rhsMemRefType = rhs.getType().cast<MemRefType>();
    auto lhsMemRefType = cast<MemRefType>(lhs.getType());
    auto rhsMemRefType = cast<MemRefType>(rhs.getType());

    // Create memref for comparison result (element type is i1)
    SmallVector<int64_t, 4> resultShape(lhsMemRefType.getShape().begin(),
                                        lhsMemRefType.getShape().end());
    auto resultMemRefType = MemRefType::get(resultShape, rewriter.getI1Type());
    auto resultAlloc = insertAllocAndDealloc(resultMemRefType, loc, rewriter);

    // Create nested loops to compare each element
    int64_t rank = lhsMemRefType.getRank();
    SmallVector<int64_t, 4> lowerBounds(rank, 0);
    SmallVector<int64_t, 4> steps(rank, 1);
    
    // Build nested loops in lowerOpToLoops style
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, lhsMemRefType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Load current elements from lhs and rhs
          auto loadedLhs = nestedBuilder.create<affine::AffineLoadOp>(loc, lhs, ivs);
          auto loadedRhs = nestedBuilder.create<affine::AffineLoadOp>(loc, rhs, ivs);
          
          // Perform element-wise comparison
          auto cmpResult = nestedBuilder.create<arith::CmpFOp>(
              loc, rewriter.getI1Type(), pred, loadedLhs, loadedRhs);
          
          // Store comparison result
          nestedBuilder.create<affine::AffineStoreOp>(loc, cmpResult, resultAlloc, ivs);
        });

    // Replace original operation with result memref
    rewriter.replaceOp(op, resultAlloc);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ReshapeOpLowering
//===----------------------------------------------------------------------===//
struct ReshapeOpLowering : public ConversionPattern {
  ReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::ReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    riscv::ReshapeOpAdaptor adaptor(operands);
    riscv::ReshapeOp reshapeOp = llvm::cast<riscv::ReshapeOp>(op);

    Value src = adaptor.getSource();
    Value shape = adaptor.getShape();
    auto resultType = dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
    // auto resultType = reshapeOp.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    auto memrefType = mlir::MemRefType::get(resultType.getShape(),
                                            resultType.getElementType());
    auto newView = rewriter.create<mlir::memref::ReshapeOp>(
        loc, memrefType, src, shape);
    rewriter.replaceOp(op, newView.getResult());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Conv2D operations
//===----------------------------------------------------------------------===//

struct Conv2DOpLowering : public ConversionPattern {
  Conv2DOpLowering(MLIRContext *ctx)
      : ConversionPattern(riscv::Conv2DOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    // output mem alloc and dealloc
    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    auto outputMem = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMem, loc, rewriter);

    riscv::Conv2DOpAdaptor conv2dAdaptor(operands);
    Value input = conv2dAdaptor.getInput();
    Value kernel = conv2dAdaptor.getKernel();
    // Value bias = conv2dAdaptor.getBias();

    // ranked tensor type
    auto inputType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto kernelType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());

    Type elemType = inputType.getElementType(); 

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();

    // input layout
    int64_t IH = inputShape[0];
    int64_t IW = inputShape[1];

    // kernel layout
    int64_t KH = kernelShape[0];
    int64_t KW = kernelShape[1];

    // output layout
    ArrayRef<int64_t> outputShape = output.getShape();
    int64_t OH = outputShape[0];
    int64_t OW = outputShape[1];

    AffineExpr d0, d1, d2, d3; // declare affine expression: i, j, p, q
    bindDims(
        rewriter.getContext(), d0, d1, d2,
        d3); // bind affine expr d0, d1 to current input dimension i, j, p, q

    // input affine map
    AffineMap inputMap = AffineMap::get(
        4, 0, ArrayRef<AffineExpr>{d0 + d2, d1 + d3}, rewriter.getContext());
    // kernel affine map
    AffineMap kernelMap = AffineMap::get(4, 0, ArrayRef<AffineExpr>{d2, d3},
                                         rewriter.getContext());

    // loops
    int64_t lb = 0, step = 1;
    /* looping i*/
    affine::AffineForOp forOpI = rewriter.create<affine::AffineForOp>(loc, lb, OH, step);
    rewriter.setInsertionPointToStart(forOpI.getBody());
    auto ivI = forOpI.getInductionVar();

    /* looping j*/
    affine::AffineForOp forOpJ = rewriter.create<affine::AffineForOp>(loc, lb, OW, step);
    rewriter.setInsertionPointToStart(forOpJ.getBody());
    auto ivJ = forOpJ.getInductionVar();

    // initilize output val
    Value zeroVal;
    // if (elemType.isa<FloatType>()) {
    if (isa<FloatType>(elemType)) {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getZeroAttr(elemType));
    } else if (isa<IntegerType>(elemType)) {
    // } else if (elemType.isa<IntegerType>()) {
      zeroVal = rewriter.create<arith::ConstantOp>(
          loc, elemType, rewriter.getIntegerAttr(elemType, 0));
    }
    
    rewriter.create<affine::AffineStoreOp>(loc, zeroVal, alloc, ValueRange{ivI, ivJ});
    /* looping p*/
    affine::AffineForOp forOpP = rewriter.create<affine::AffineForOp>(loc, lb, KH, step);
    rewriter.setInsertionPointToStart(forOpP.getBody());
    auto ivP = forOpP.getInductionVar();

    /* looping q*/
    affine::AffineForOp forOpQ = rewriter.create<affine::AffineForOp>(loc, lb, KW, step);
    rewriter.setInsertionPointToStart(forOpQ.getBody());
    auto ivQ = forOpQ.getInductionVar();

    // input bound check
    Value inputRow = rewriter.create<affine::AffineApplyOp>(
        loc, inputMap.getSubMap(0), ValueRange{ivI, ivJ, ivP, ivQ});
    Value inputCol = rewriter.create<affine::AffineApplyOp>(
        loc, inputMap.getSubMap(1), ValueRange{ivI, ivJ, ivP, ivQ});
    Value rowUB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inputRow,
        rewriter.create<arith::ConstantIndexOp>(loc, IH));
    Value colUB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inputCol,
        rewriter.create<arith::ConstantIndexOp>(loc, IW));
    Value bound = rewriter.create<arith::AndIOp>(loc, rowUB, colUB);

    // bound condition
    rewriter.create<scf::IfOp>(
        loc, bound, [&](OpBuilder &builder, Location loc) {
          // load input
          Value inputVal = builder.create<affine::AffineLoadOp>(
              loc, input, inputMap, ValueRange{ivI, ivJ, ivP, ivQ});
          Value kernelVal = builder.create<affine::AffineLoadOp>(
              loc, kernel, kernelMap, ValueRange{ivI, ivJ, ivP, ivQ});
          // mul
          Value prod;
          // if (elemType.isa<FloatType>()) {
          if (isa<FloatType>(elemType)) {
            prod = builder.create<arith::MulFOp>(loc, inputVal, kernelVal); 
          } else {
            prod = builder.create<arith::MulIOp>(loc, inputVal, kernelVal); 
          }

          Value outputVal =
              builder.create<affine::AffineLoadOp>(loc, alloc, ValueRange{ivI, ivJ});
          Value sum;
          // if (elemType.isa<FloatType>()) {
          if (isa<FloatType>(elemType)) {
            sum = builder.create<arith::AddFOp>(loc, prod, outputVal); 
          } else {
            sum = builder.create<arith::AddIOp>(loc, prod, outputVal); 
          }

          // store the computed output
          builder.create<affine::AffineStoreOp>(loc, sum, alloc, ValueRange{ivI, ivJ});

          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) { // 补全 else 分支
          builder.create<scf::YieldOp>(loc);
        });
         
    // rewriter.setInsertionPointAfter(forOpQ);
    // rewriter.setInsertionPointAfter(forOpP);
    // //add bias
    // Value convSum = rewriter.create<affine::AffineLoadOp>(loc, alloc, ValueRange{ivI, ivJ});
    // Value biasVal = rewriter.create<affine::AffineLoadOp>(loc, bias, ValueRange{});
    // Value finalResult = rewriter.create<arith::AddFOp>(loc, convSum, biasVal);
    // rewriter.create<affine::AffineStoreOp>(loc, finalResult, alloc, ValueRange{ivI, ivJ});
    // rewriter.create<affine::AffineStoreOp>(loc, finalResult, alloc, ValueRange{ivI, ivJ});

    rewriter.replaceOp(op, alloc);

    return success();
  }
}; // conv2d

//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Launch operations
//===----------------------------------------------------------------------===//

class LaunchOpLowering : public ConversionPattern {
public:
  explicit LaunchOpLowering(MLIRContext *context)
      : ConversionPattern(riscv::LaunchOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    riscv::LaunchOp launch = cast<riscv::LaunchOp>(op);

    std::string launch_name("launch");
    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      launch_name = attr.getValue().str();

    SmallVector<Value> lbs, ubs, steps;
    auto c0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);

    // make scf.parallel to replace riscv.launch
    for (auto d : launch.getSizeOperands()) {
      lbs.push_back(c0);
      ubs.push_back(d);
      steps.push_back(c1);
    }
    if (lbs.empty()) {
      lbs.push_back(c0);
      ubs.push_back(c1);
      steps.push_back(c1);
    }
    auto scfPar =
        rewriter.create<scf::ParallelOp>(op->getLoc(), lbs, ubs, steps);

    IRMapping remap;

    // map launch iteration space to scf.parallel ivs
    for (auto p : llvm::zip(launch.getIds(), scfPar.getInductionVars()))
      remap.map(std::get<0>(p), std::get<1>(p));

    // map launch size to scf.parallel upper bounds
    for (auto p : llvm::zip(launch.getSizeOperands(), scfPar.getUpperBound()))
      remap.map(std::get<0>(p), std::get<1>(p));

    // remap isolated from above launch operands
    auto launchOperands =
        operands.drop_front(operands.size() - launch.getNumKernelOperands());
    for (auto p : llvm::zip(launch.getKernelArguments(), launchOperands))
      remap.map(std::get<0>(p), std::get<1>(p));

    // clone the body
    rewriter.setInsertionPointToStart(scfPar.getBody());
    auto &launchOps = launch.getBody().front().getOperations();
    for (auto bi = launchOps.begin(), be = --launchOps.end(); bi != be; ++bi)
      rewriter.clone(*bi, remap);

    // replace output events with riscv.wait_all
    if (op->getNumResults()) {
      SmallVector<Value> deps;
      for (auto &o : operands)
        if (llvm::isa<riscv::EventType>(o.getType()))
          deps.push_back(o);
      rewriter.setInsertionPoint(scfPar);
      rewriter.replaceOpWithNewOp<riscv::WaitAllOp>(
          op, riscv::EventType::get(op->getContext()), deps);
    } else
      rewriter.eraseOp(launch);

    return success();
  }
};


//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Herd operations
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Load operations
//===----------------------------------------------------------------------===//
LogicalResult lowerLoad(Operation *op, PatternRewriter &rewriter,
                                std::string fnName) {
  auto ctx = op->getContext();
  auto loc = op->getLoc();

  auto dmaOp = cast<riscv::LoadDrvOp>(op); 
  SmallVector<Type, 6> tys;
  SmallVector<Value, 16> operands;

  SmallVector<Type, 1> retTys(op->getNumResults(), LLVM::LLVMPointerType::get(ctx));
  auto i32Ty = IntegerType::get(ctx, 32);
  auto signalTy = LLVM::LLVMPointerType::get(ctx);
  if (op->getNumResults()) {
    // 异步拷贝：创建信号变量
    auto one = rewriter.create<arith::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
    auto signal = rewriter.create<LLVM::AllocaOp>(loc, signalTy, i32Ty, one, 4);
    operands.push_back(signal);
  } else {
    // 同步拷贝：传入空指针
    auto nullV = rewriter.create<LLVM::ZeroOp>(loc, signalTy).getResult();
    operands.push_back(nullV);
  }

  MemRefType dstMemRefTy = llvm::cast<MemRefType>(dmaOp.getDstMemref().getType());
  MemRefType srcMemRefTy = llvm::cast<MemRefType>(dmaOp.getSrcMemref().getType());
  
  // 强制转换为动态维度（适配底层函数调用）
  operands.push_back(rewriter.create<memref::CastOp>(
      loc,
      MemRefType::get(
          std::vector<int64_t>(dstMemRefTy.getRank(), ShapedType::kDynamic),
          dstMemRefTy.getElementType(), dstMemRefTy.getLayout(),
          dstMemRefTy.getMemorySpace()),
      dmaOp.getDstMemref()));
  operands.push_back(rewriter.create<memref::CastOp>(
      loc,
      MemRefType::get(
          std::vector<int64_t>(srcMemRefTy.getRank(), ShapedType::kDynamic),
          srcMemRefTy.getElementType(), srcMemRefTy.getLayout(),
          srcMemRefTy.getMemorySpace()),
      dmaOp.getSrcMemref()));

  auto i64Ty = rewriter.getI64Type();
  auto zero = rewriter.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
  auto one = rewriter.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 1));
  
  SmallVector<Value, 4> offsets(4, zero);
  SmallVector<Value, 4> lengths(4, one);
  SmallVector<Value, 3> strides(3, zero);

  // 填充源偏移（仅处理LMEM/GMEM的rank适配）
  int idx = 4 - srcMemRefTy.getRank();
  for (auto o : dmaOp.getSrcOffsets())
    offsets[idx++] = rewriter.create<arith::IndexCastOp>(loc, i64Ty, o);
  
  // 填充源长度
  idx = 4 - srcMemRefTy.getRank();
  for (auto o : dmaOp.getSrcSizes())
    lengths[idx++] = rewriter.create<arith::IndexCastOp>(loc, i64Ty, o);
  
  // 填充源步长
  idx = 4 - dstMemRefTy.getRank();
  if (!dmaOp.getSrcStrides().empty())
    for (auto o : dmaOp.getSrcStrides().drop_back())
      strides[idx++] = rewriter.create<arith::IndexCastOp>(loc, i64Ty, o);

  // 追加偏移/长度/步长到参数列表
  operands.append(offsets.begin(), offsets.end());
  operands.append(lengths.begin(), lengths.end());
  operands.append(strides.begin(), strides.end());

  // 构造函数名（按内存空间+类型拼接，区分LMEM/GMEM）
  for (auto o : operands)
    tys.push_back(o.getType());
  llvm::raw_string_ostream ss(fnName);
  ss << "_" << dstMemRefTy.getRank() << "d" << dstMemRefTy.getMemorySpaceAsInt();
  dstMemRefTy.getElementType().print(ss);
  ss << "_" << srcMemRefTy.getRank() << "d" << srcMemRefTy.getMemorySpaceAsInt();
  srcMemRefTy.getElementType().print(ss);

  // 创建/查找底层函数，生成调用
  auto module = op->getParentOfType<ModuleOp>();
  auto fn = module.lookupSymbol<func::FuncOp>(ss.str());
  if (!fn) {
    auto fnTy = FunctionType::get(ctx, tys, retTys);
    fn = func::FuncOp::create(rewriter.getUnknownLoc(), ss.str(), fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  auto call = rewriter.create<func::CallOp>(op->getLoc(), retTys,
                                            SymbolRefAttr::get(fn), operands);
  if (op->getNumResults()) {
    rewriter.replaceOp(op, call.getResults());
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns:GMEMAllocOpConversion
//===----------------------------------------------------------------------===//
class GMEMAllocOpConversion : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getType();
    if (op.getType().getMemorySpaceAsInt() != (int)riscv::MemorySpace::GMEM)
      return failure();

    auto alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(),
        MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                        memrefTy.getLayout(), 0));
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns:GMEMDeallocOpConversion
//===----------------------------------------------------------------------===//
class GMEMDeallocOpConversion : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)riscv::MemorySpace::GMEM)
      return failure();

    rewriter.create<memref::DeallocOp>(op.getLoc(), adaptor.getMemref());
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Alloc Interface
//===----------------------------------------------------------------------===//
static uint64_t getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume * (ty.getElementTypeBitWidth() / 8);
}

static uint64_t getTensorVolume(const Type ty) {
  if (auto t = llvm::dyn_cast<ShapedType>(ty)) {
    return getTensorVolume(t);
  } else {
    return 1;
  }
}
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: LMEMAllocOpConversion
//===----------------------------------------------------------------------===//
class LMEMAllocOpConversion : public OpRewritePattern<riscv::AllocOp> {
public:
  using OpRewritePattern<riscv::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(riscv::AllocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;

    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)riscv::MemorySpace::LMEM)
      return failure();

    tys.push_back(IndexType::get(ctx));
    retTys.push_back(MemRefType::get(
        std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
        memrefTy.getElementType(), memrefTy.getLayout(),
        memrefTy.getMemorySpace()));

    auto size = getTensorVolume(memrefTy);
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), size));

    auto module = op->getParentOfType<ModuleOp>();

    std::string fnName = "__npu_mem_malloc";
    llvm::raw_string_ostream ss(fnName);
    // ss << "_" << memrefTy.getRank();
    // ss << "d" << memrefTy.getMemorySpaceAsInt();
    // memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
    auto castOp = rewriter.create<memref::CastOp>(op->getLoc(), memrefTy,
                                                  callOp.getResult(0));
    rewriter.replaceOp(op, castOp->getResults());
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: Load operations
//===----------------------------------------------------------------------===//
class LMEMDeallocOpConversion
    : public OpRewritePattern<riscv::DeallocOp> {
public:
  using OpRewritePattern<riscv::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(riscv::DeallocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;
    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)riscv::MemorySpace::LMEM)
      return failure();

    tys.push_back(MemRefType::get(
        std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
        memrefTy.getElementType(), memrefTy.getLayout(),
        memrefTy.getMemorySpace()));
    operands.push_back(
        rewriter.create<memref::CastOp>(op->getLoc(), tys[0], op.getMemref()));

    auto module = op->getParentOfType<ModuleOp>();

    std::string fnName = "__npu_mem_free";
    llvm::raw_string_ostream ss(fnName);
    // ss << "_" << memrefTy.getRank();
    // ss << "d" << memrefTy.getMemorySpaceAsInt();
    // memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    rewriter.create<func::CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn),
                                  operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// General template class: for *_DRV Lowering
//===----------------------------------------------------------------------===//

template <typename OpTy, const char* IntrinsicName>
class DrvOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    
    BaseMemRefType srcMemRefTy = llvm::cast<BaseMemRefType>(op.getSrcMemref().getType());
    BaseMemRefType dstMemRefTy = llvm::cast<BaseMemRefType>(op.getDstMemref().getType());
    (void)srcMemRefTy; (void)dstMemRefTy; 

    SmallVector<Value, 4> deps;
    for (auto o : op.getOperands()) {
      if (llvm::isa<riscv::EventType>(o.getType())) {
        deps.push_back(o);
      }
    }
    if (!deps.empty()) {
      rewriter.create<riscv::WaitAllOp>(loc, riscv::EventType::get(ctx), deps);
    }

    Type i32Ty = IntegerType::get(ctx, 32);
    Value srcIndex = op.getSrcOffsets()[0];
    Value dstIndex = op.getDstOffsets()[0];
    Value srcI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, srcIndex);
    Value dstI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dstIndex);

    auto module = op->template getParentOfType<mlir::ModuleOp>();
    
    SmallVector<Type, 2> funcArgsTy = {i32Ty, i32Ty};
    SmallVector<Type, 1> funcRetTy = {i32Ty};
    auto funcType = mlir::FunctionType::get(ctx, funcArgsTy, funcRetTy);

    auto dmaFunc = module.template lookupSymbol<mlir::func::FuncOp>(IntrinsicName);
    
    if (!dmaFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      dmaFunc = mlir::func::FuncOp::create(loc, IntrinsicName, funcType);
      dmaFunc.setPrivate();
      module.push_back(dmaFunc);
    } else {
      if (dmaFunc.getFunctionType() != funcType) {
        return op->emitError("existing function '")
               << IntrinsicName << "' has incompatible type";
      }
    }

    SmallVector<Value, 2> callArgs = {srcI32, dstI32};
    SmallVector<Type, 1> callResultTypes = {i32Ty};
    rewriter.create<mlir::func::CallOp>(loc, callResultTypes,
                                  mlir::SymbolRefAttr::get(ctx, IntrinsicName),
                                  callArgs);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: TLD_DRV operations
//===----------------------------------------------------------------------===//
const char LoadIntrinsicName[] = "llvm.riscv.load";
using LoadOpLowering = DrvOpLowering<riscv::LoadDrvOp, LoadIntrinsicName>;

//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: TST_DRV operations
//===----------------------------------------------------------------------===//
const char StoreIntrinsicName[] = "llvm.riscv.store"; 
using StoreOpLowering = DrvOpLowering<riscv::StoreDrvOp, StoreIntrinsicName>;


//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: WaitAll operations
//===----------------------------------------------------------------------===//

// struct WaitAllOpLowering
//     : public OpConversionPattern<riscv::WaitAllOp> {
// public:
//   using OpConversionPattern<riscv::WaitAllOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(riscv::WaitAllOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     SmallVector<Value, 8> operands{adaptor.getOperands()};
//     auto module = op->getParentOfType<ModuleOp>();
//     auto ctx = op->getContext();

//     SmallVector<Type, 8> tys(operands.size(), LLVM::LLVMPointerType::get(ctx));
//     SmallVector<Type, 1> retTys(op->getNumResults(),
//                                 LLVM::LLVMPointerType::get(ctx));

//     std::string fnName = "__cim_wait_all";
//     llvm::raw_string_ostream ss(fnName);
//     ss << "_" << retTys.size() << "_" << operands.size();

//     auto fn = module.lookupSymbol<func::FuncOp>(fnName);
//     if (!fn) {
//       auto fnTy = FunctionType::get(ctx, tys, retTys);
//       fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
//       fn.setPrivate();
//       module.push_back(fn);
//     }

//     rewriter.replaceOpWithNewOp<func::CallOp>(op, retTys,
//                                               SymbolRefAttr::get(fn), operands);
//     return success();
//   }
// };

class WaitAllOpLowering : public OpConversionPattern<riscv::WaitAllOp> {
public:
  using OpConversionPattern<riscv::WaitAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(riscv::WaitAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op->getContext();

    // 定义函数参数/返回值类型（统一为 LLVM 指针类型）
    SmallVector<Type, 8> tys(operands.size(), mlir::LLVM::LLVMPointerType::get(ctx));
    SmallVector<Type, 1> retTys(op->getNumResults(), mlir::LLVM::LLVMPointerType::get(ctx));

    std::string fnName = "llvm.riscv.sync";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << retTys.size() << "_" << operands.size();

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = mlir::FunctionType::get(ctx, tys, retTys);
      fn = mlir::func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate(); // 标记为私有函数
      module.push_back(fn);
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op,                              // 被替换的 Op
        retTys,                          // 返回值类型
        mlir::SymbolRefAttr::get(fn),    // 调用的函数名
        operands                         // 函数参数
    );

    return mlir::success();
  }
};
//===----------------------------------------------------------------------===//
// ToAffine RewritePatterns: ScfParOpConversion
//===----------------------------------------------------------------------===//
LogicalResult ScfParToAffineForConversion(Operation *op) {
  func::FuncOp f = dyn_cast<func::FuncOp>(op);
  if (!f)
    return failure();

  llvm::SetVector<Operation *> erased;
  SmallVector<scf::ParallelOp> scf_pars;
  f.walk([&](scf::ParallelOp scf_par) { scf_pars.push_back(scf_par); });
  for (auto scf_par : scf_pars) {
    if (!llvm::all_of(scf_par.getLowerBound(), [](Value v) {
          auto constV = getConstantIntValue(v);
          if (!constV)
            return false;
          if (*constV != 0)
            return false;
          return true;
        })) {
      scf_par->emitOpError("has non-zero lower bound.");
      return failure();
    }
    if (!llvm::all_of(scf_par.getStep(), [](Value v) {
          auto constV = getConstantIntValue(v);
          if (!constV)
            return false;
          if (*constV != 1)
            return false;
          return true;
        })) {
      scf_par->emitOpError("has non-unit step size.");
      return failure();
    }
    std::vector<int> par_sizes = {};
    for (auto v : scf_par.getUpperBound())
      par_sizes.push_back(
          dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp()).value());

    OpBuilder builder(scf_par);
    SmallVector<affine::AffineForOp> loops;
    for (unsigned i = 0; i < par_sizes.size(); i++) {
      if (i == 0)
        loops.push_back(builder.create<affine::AffineForOp>(scf_par.getLoc(), 0,
                                                            par_sizes[0]));
      else {
        auto inner_builder = OpBuilder::atBlockBegin(loops[i - 1].getBody());
        loops.push_back(inner_builder.create<affine::AffineForOp>(
            scf_par.getLoc(), 0, par_sizes[i]));
      }
    }

    builder.setInsertionPointToStart(loops.back().getBody());
    IRMapping remap;
    for (unsigned i = 0; i < par_sizes.size(); i++)
      remap.map(scf_par.getInductionVars()[i], loops[i].getInductionVar());
    for (auto &o : scf_par.getBody()->getOperations()) {
      if (!isa<scf::ReduceOp>(o) && !isa<scf::YieldOp>(o) &&
          !isa<scf::ParallelOp>(o)) {
        builder.clone(o, remap);
      }
    }
    erased.insert(scf_par);
  }
  for (auto a : erased) {
    if (a->getNumResults())
      for (auto token : a->getResults())
        for (auto user : token.getUsers())
          for (unsigned i = 0; i < user->getNumOperands(); i++)
            if (user->getOperand(i) == token)
              user->eraseOperand(i);
    a->erase();
  }
  return success();
}

namespace {
class RISCVToAffineLowerPass
    : public mlir::PassWrapper<RISCVToAffineLowerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-riscv-to-affine"; 
  }
  StringRef getDescription() const final {
    return "Lower RISCV dialect operations to Affine dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RISCVToAffineLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void RISCVToAffineLowerPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<riscv::RISCVDialect>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  target.addDynamicallyLegalOp<riscv::PrintOp>([](riscv::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
      return mlir::isa<mlir::TensorType>(type);
    });
  });
  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>([](mlir::arith::ConstantOp op) {
    return !mlir::isa<mlir::TensorType>(op.getType());
  });
  target.addLegalOp<riscv::WorldOp>();

    // Add type converter
  mlir::TypeConverter typeConverter;
  typeConverter.addConversion([](mlir::Type type) { return type; });
  typeConverter.addConversion([](mlir::TensorType type) -> mlir::Type {
    return convertTensorToMemRef(type);
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ScalarConstantOpLowering,TensorConstantOpLowering,ReduceOpLowering,
  MatmulOpLowering,MatvecOpLowering,TransposeOpLowering,ReshapeOpLowering,
  CmpIOpLowering,CmpFOpLowering,AddOpLowering,SubOpLowering,MulOpLowering,DivOpLowering,
  MaxOpLowering,MinOpLowering,AndIOpLowering,XOrIOpLowering,OrIOpLowering,NegFOpLowering,
  Conv2DOpLowering,WaitAllOpLowering,LaunchOpLowering,
  LoadOpLowering, StoreOpLowering,
  GMEMAllocOpConversion,GMEMDeallocOpConversion,
  LMEMAllocOpConversion,LMEMDeallocOpConversion>(&getContext());
  
  patterns.add<PrintOpLowering, ArithConstantOpLowering>(typeConverter, &getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }

    module.walk([&](mlir::func::FuncOp f) {
    if (failed(ScfParToAffineForConversion(f))) {
      mlir::emitError(mlir::UnknownLoc::get(context), "error lowering scf.parallel to affine.for");
      signalPassFailure();
    }
  });

}


namespace riscv{
  std::unique_ptr<mlir::Pass> createLowerToAffinePass() {
    return std::make_unique<RISCVToAffineLowerPass>();
  }
}

