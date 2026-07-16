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

#include "mlir/Dialect/Async/IR/Async.h"
#include "torch-mlir/Dialect/CIM/IR/CIMDialect.h"
#include "torch-mlir/Dialect/CIM/IR/CIMOps.h"
#include "torch-mlir/Conversion/RISCVPasses.h"
#include "torch-mlir/Dialect/RISCV/IR/RISCVDialect.h"
#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.h"
// #include "torch-mlir/Conversion/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include <iostream>


using namespace mlir;
//===----------------------------------------------------------------------===//
// 辅助函数
//===----------------------------------------------------------------------===//
Value extractMemRefBaseBufferToLLVMPtr(Value memrefVal,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        Type llvmPtrType)  {
  MemRefType memRefType = cast<MemRefType>(memrefVal.getType());
  Type indexType = rewriter.getIndexType();
  unsigned memRefRank = memRefType.getRank();
  Type i64Type = rewriter.getI64Type();
  Type i32Type = rewriter.getI32Type(); 

  SmallVector<Type> resultTypes;
  resultTypes.push_back(MemRefType::get({}, memRefType.getElementType()));
  resultTypes.push_back(indexType);
  for (unsigned i = 0; i < memRefRank; ++i) {
    resultTypes.push_back(indexType);
  }
  for (unsigned i = 0; i < memRefRank; ++i) {
    resultTypes.push_back(indexType);
  }

  auto stridedMetadataOp = rewriter.create<memref::ExtractStridedMetadataOp>(
      loc,
      resultTypes,
      memrefVal
  );
  Value baseBuffer = stridedMetadataOp.getResult(0);

  SmallVector<Value> zeroIndices;
  for (unsigned i = 0; i < cast<MemRefType>(baseBuffer.getType()).getRank(); ++ i) {
    zeroIndices.push_back(rewriter.create<arith::ConstantOp>(
        loc,
        rewriter.getIndexAttr(0)
    ).getResult());
  }

  Value bufferScalarValue = rewriter.create<memref::LoadOp>(
      loc,
      baseBuffer,
      zeroIndices
  ).getResult();

  if (!bufferScalarValue.getType().isInteger(64)) {
    if (bufferScalarValue.getType().isF32()) {
      // f32→i32（合法bitcast）→i64（扩展）
      bufferScalarValue = rewriter.create<arith::BitcastOp>(loc, i32Type, bufferScalarValue);
      bufferScalarValue = rewriter.create<arith::ExtUIOp>(loc, i64Type, bufferScalarValue);
    } else {
      bufferScalarValue = rewriter.create<arith::BitcastOp>(loc, i64Type, bufferScalarValue);
    }
  }

  Value llvmPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrType, bufferScalarValue);
  if (llvmPtr.getType() != llvmPtrType) {
    llvmPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, llvmPtrType, llvmPtr);
  }
  return llvmPtr;
}

// 提取memref底层缓冲区并转为uint32_t
Value extractMemRefBaseBufferToUint32(Value memrefVal,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        Type uint32Type) {
  MemRefType memRefType = cast<MemRefType>(memrefVal.getType());
  Type indexType = rewriter.getIndexType();
  unsigned memRefRank = memRefType.getRank();
  Type i64Type = rewriter.getI64Type(); 
  Type i32Type = rewriter.getI32Type(); 

  SmallVector<Type> resultTypes;
  resultTypes.push_back(MemRefType::get({}, memRefType.getElementType()));
  resultTypes.push_back(indexType);
  for (unsigned i = 0; i < memRefRank; ++i) {
    resultTypes.push_back(indexType);
  }
  for (unsigned i = 0; i < memRefRank; ++i) {
    resultTypes.push_back(indexType);
  }

  auto stridedMetadataOp = rewriter.create<memref::ExtractStridedMetadataOp>(
      loc,
      resultTypes,
      memrefVal
  );
  Value baseBuffer = stridedMetadataOp.getResult(0);

  SmallVector<Value> zeroIndices;
  for (unsigned i = 0; i < cast<MemRefType>(baseBuffer.getType()).getRank(); ++i) {
    zeroIndices.push_back(rewriter.create<arith::ConstantOp>(
        loc,
        rewriter.getIndexAttr(0)
    ).getResult());
  }

  Value bufferScalarValue = rewriter.create<memref::LoadOp>(loc, baseBuffer, zeroIndices).getResult();
  
  if (!bufferScalarValue.getType().isInteger(64)) {
    if (bufferScalarValue.getType().isF32()) {
      // f32→i32（合法bitcast）→i64（扩展）
      bufferScalarValue = rewriter.create<arith::BitcastOp>(loc, i32Type, bufferScalarValue);
      bufferScalarValue = rewriter.create<arith::ExtUIOp>(loc, i64Type, bufferScalarValue);
    } else {
      bufferScalarValue = rewriter.create<arith::BitcastOp>(loc, i64Type, bufferScalarValue);
    }
  }

  Value bufferUint32 = rewriter.create<arith::TruncIOp>(loc, uint32Type, bufferScalarValue).getResult();
  if (bufferUint32.getType() != uint32Type) {
    bufferUint32 = rewriter.create<arith::BitcastOp>(loc, uint32Type, bufferUint32);
  }
  return bufferUint32;
}

//从uint32_t地址构建合法 memref
Value buildValidMemRefFromUint32(Value bufferUint32,
                                  Type memrefType,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  
  MemRefType targetMemRefType = cast<MemRefType>(memrefType);
  SmallVector<Value> dynamicDims;
  Type i64Type = rewriter.getI64Type();
  Type llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext()); 
  //分配新的memref内存空间
  Value newMemRef = rewriter.create<memref::AllocOp>(
      loc,
      targetMemRefType,
      dynamicDims
  ).getResult();
  //i32->i64（无符号扩展，保留 uint32_t 完整值）
  Value bufferInt64 = rewriter.create<arith::ExtUIOp>(
      loc,
      i64Type,
      bufferUint32
  ).getResult();
  Value bufferPtr = rewriter.create<LLVM::IntToPtrOp>(
      loc,
      llvmPtrType,
      bufferInt64
  ).getResult();

  Value newMemRefBuffer = extractMemRefBaseBufferToLLVMPtr(newMemRef, rewriter, loc, llvmPtrType);
  rewriter.create<LLVM::StoreOp>(
      loc,
      bufferPtr,
      newMemRefBuffer,
      /*isVolatile=*/false
  );

  return newMemRef;
}

//===----------------------------------------------------------------------===//
// RISCVToLLVMCall RewritePatterns: MatMulOpLoweringToCIM
//===----------------------------------------------------------------------===//
class MatMulOpLoweringToCIM
    : public OpConversionPattern<rair::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(rair::MatmulOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value out = adaptor.getOperands()[2];
  
    MLIRContext *ctx = rewriter.getContext();
    Type uint32Type = rewriter.getI32Type(); 
    // auto lhsMemRefType = cast<MemRefType>(lhs.getType());

    // ========== 提取memref<>底层地址并转为uint32_t（i32）==========
    Value lhsUint32 = extractMemRefBaseBufferToUint32(lhs, rewriter, op.getLoc(), uint32Type);
    Value rhsUint32 = extractMemRefBaseBufferToUint32(rhs, rewriter, op.getLoc(), uint32Type);

    SmallVector<Value, 2> callOperands = {lhsUint32, rhsUint32};

    auto module = op->getParentOfType<ModuleOp>();

    SmallVector<Type, 2> inputTypes = {uint32Type, uint32Type};
    auto fnType = FunctionType::get(ctx, inputTypes, {uint32Type});
    constexpr const char *kFuncName = "llvm.riscv.vv.v.drv";

    func::FuncOp func = module.lookupSymbol<func::FuncOp>(kFuncName);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      func = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), kFuncName, fnType);
      func.setPrivate();
    }

    auto callOp = rewriter.create<func::CallOp>(
        op.getLoc(),
        func,
        callOperands);
    Value resultUint32 = callOp.getResult(0); 

    if (out) {
      Value resultMemRef = buildValidMemRefFromUint32(resultUint32, out.getType(), rewriter, op.getLoc());
      rewriter.create<memref::CopyOp>(op.getLoc(), resultMemRef, out);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// RISCVToLLVMCall RewritePatterns: BatchMatmul(ConvDrv) operations
//===----------------------------------------------------------------------===//
// class BatchMatMulOpLoweringToCIM
//     : public OpConversionPattern<rair::BatchMatMulOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(rair::BatchMatMulOp op,
//                                 OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Value lhs = adaptor.getOperands()[0];
//     Value rhs = adaptor.getOperands()[1];
  
//     SmallVector<Value, 2> callOperands = {lhs, rhs};

//     auto module = op->getParentOfType<ModuleOp>();
//     MLIRContext *ctx = rewriter.getContext();

//     SmallVector<Type, 2> inputTypes = {
//         lhs.getType(),
//         rhs.getType()
//     };

//     auto fnType = FunctionType::get(ctx, inputTypes, {});
//     constexpr const char *kFuncName = "llvm.riscv.conv.drv";

//     func::FuncOp func = module.lookupSymbol<func::FuncOp>(kFuncName);
//     if (!func) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module.getBody());

//       func = rewriter.create<func::FuncOp>(
//           rewriter.getUnknownLoc(), kFuncName, fnType);
//       func.setPrivate();
//     }
//     rewriter.create<func::CallOp>(
//         op.getLoc(),
//         func,
//         callOperands);
//     rewriter.eraseOp(op);
//     return success();
//   }
// };
//===----------------------------------------------------------------------===//
// RISCVToLLVMCall RewritePatterns: BatchMatmul(ConvDrv) operations
//===----------------------------------------------------------------------===//
class BatchMatMulOpLoweringToCIM
    : public OpConversionPattern<rair::BatchMatMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(rair::BatchMatMulOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 提取BatchMatmul的输入操作数（lhs/rhs）和输出（out）
    // 注：需确认rair::BatchMatMulOp的operands顺序，通常是[lhs, rhs, out]（和Matmul一致）
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value out = adaptor.getOperands().size() > 2 ? adaptor.getOperands()[2] : nullptr;
  
    MLIRContext *ctx = rewriter.getContext();
    Type uint32Type = rewriter.getI32Type(); 

    // 2. 核心：提取memref底层地址转为uint32_t（和Matmul逻辑完全一致）
    Value lhsUint32 = extractMemRefBaseBufferToUint32(lhs, rewriter, op.getLoc(), uint32Type);
    Value rhsUint32 = extractMemRefBaseBufferToUint32(rhs, rewriter, op.getLoc(), uint32Type);

    // 3. 构造调用参数（uint32_t类型的lhs/rhs地址）
    SmallVector<Value, 2> callOperands = {lhsUint32, rhsUint32};

    auto module = op->getParentOfType<ModuleOp>();

    // 4. 定义函数类型（输入：两个uint32_t，输出：一个uint32_t，和Matmul一致）
    SmallVector<Type, 2> inputTypes = {uint32Type, uint32Type};
    auto fnType = FunctionType::get(ctx, inputTypes, {uint32Type});
    constexpr const char *kFuncName = "llvm.riscv.conv.drv";

    // 5. 查找/创建自定义函数（逻辑和Matmul完全一致）
    func::FuncOp func = module.lookupSymbol<func::FuncOp>(kFuncName);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      func = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), kFuncName, fnType);
      func.setPrivate();
    }

    // 6. 调用自定义函数并获取返回结果（uint32_t类型的结果地址）
    auto callOp = rewriter.create<func::CallOp>(
        op.getLoc(),
        func,
        callOperands);
    Value resultUint32 = callOp.getResult(0); 

    // 7. 若有输出out，构建memref并拷贝结果（和Matmul逻辑完全一致）
    if (out) {
      Value resultMemRef = buildValidMemRefFromUint32(resultUint32, out.getType(), rewriter, op.getLoc());
      rewriter.create<memref::CopyOp>(op.getLoc(), resultMemRef, out);
    }

    // 8. 删除原BatchMatmul Op
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// TransposeOpLoweringToCIM
//===----------------------------------------------------------------------===//

struct TransposeOpLoweringToCIM : public ConversionPattern {
  TransposeOpLoweringToCIM(MLIRContext *ctx)
      : ConversionPattern(rair::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto transposeOp = cast<rair::TransposeOp>(op);

    Value inputMemRef = transposeOp.getInput();   
    Value outputMemRef = transposeOp.getInit();   
    if (!inputMemRef || !outputMemRef) {
      return rewriter.notifyMatchFailure(loc, "transpose op missing input/output memref");
    }

    Type uint32Type = rewriter.getI32Type(); // 对应uint32_t
    // 提取输入MemRef的底层地址（转为uint32_t）
    Value inputUint32 = extractMemRefBaseBufferToUint32(
        inputMemRef, rewriter, loc, uint32Type);
    if (!inputUint32) {
      return rewriter.notifyMatchFailure(loc, "failed to extract input memref address");
    }

    MLIRContext *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module) {
      return rewriter.notifyMatchFailure(loc, "transpose op not in a module");
    }

    SmallVector<Type, 1> inputTypes = {uint32Type};
    auto fnType = FunctionType::get(ctx, inputTypes, {uint32Type});
    constexpr const char *kTransFuncName = "llvm.riscv.trans.drv";

    // 查找或创建llvm.riscv.trans.drv函数
    func::FuncOp transFunc = module.lookupSymbol<func::FuncOp>(kTransFuncName);
    if (!transFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      transFunc = rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), kTransFuncName, fnType);
      transFunc.setPrivate();
    }

    SmallVector<Value, 1> callOperands = {inputUint32}; 
    auto callOp = rewriter.create<func::CallOp>(
        loc, transFunc, callOperands);
    Value outputUint32 = callOp.getResult(0); 

    if (outputMemRef) {
      Value resultMemRef = buildValidMemRefFromUint32(
          outputUint32, outputMemRef.getType(), rewriter, loc);
      if (!resultMemRef) {
        return rewriter.notifyMatchFailure(loc, "failed to build output memref from address");
      }

      rewriter.create<memref::CopyOp>(loc, resultMemRef, outputMemRef);
    }

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
      if (llvm::isa<rair::EventType>(o.getType())) {
        deps.push_back(o);
      }
    }
    if (!deps.empty()) {
      rewriter.create<rair::WaitAllOp>(loc, rair::EventType::get(ctx), deps);
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
using LoadOpLowering = DrvOpLowering<rair::LoadDrvOp, LoadIntrinsicName>;

//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: TST_DRV operations 
//===----------------------------------------------------------------------===//
const char StoreIntrinsicName[] = "llvm.riscv.store"; 
using StoreOpLowering = DrvOpLowering<rair::StoreDrvOp, StoreIntrinsicName>;

// } // namespace rair

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
class LMEMAllocOpConversion : public OpRewritePattern<rair::AllocOp> {
public:
  using OpRewritePattern<rair::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rair::AllocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;

    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)rair::MemorySpace::LMEM)
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
// ToAffine RewritePatterns: DeallocOp operations
//===----------------------------------------------------------------------===//
class LMEMDeallocOpConversion
    : public OpRewritePattern<rair::DeallocOp> {
public:
  using OpRewritePattern<rair::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rair::DeallocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;
    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)rair::MemorySpace::LMEM)
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
// ToLLVMFunc RewritePatterns: WaitAll operations
//===----------------------------------------------------------------------===//

class WaitAllOpLowering : public OpConversionPattern<rair::WaitAllOp> {
public:
  using OpConversionPattern<rair::WaitAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(rair::WaitAllOp op, OpAdaptor adaptor,
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
// ToLLVMFunc RewritePatterns: PoolingNCHWMaxOp operations
//===----------------------------------------------------------------------===//
class PoolingNCHWMaxOpLoweringToCIM
    : public OpConversionPattern<rair::PoolingNchwMaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(rair::PoolingNchwMaxOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    Value kernel = adaptor.getOperands()[1];
    Value output = adaptor.getOperands()[2];
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type uint32Type = rewriter.getI32Type();

    Value inputUint32 = extractMemRefBaseBufferToUint32(input, rewriter, loc, uint32Type);
    Value kernelUint32 = extractMemRefBaseBufferToUint32(kernel, rewriter, loc, uint32Type);
    Value outputUint32 = extractMemRefBaseBufferToUint32(output, rewriter, loc, uint32Type);

    auto module = op->getParentOfType<ModuleOp>();

    SmallVector<Value, 2> vvVDrv1Operands = {inputUint32, kernelUint32};
    SmallVector<Type, 2> vvVDrv1InputTypes = {uint32Type, uint32Type};
    auto vvVDrv1FnType = FunctionType::get(ctx, vvVDrv1InputTypes, {uint32Type});
    constexpr const char *kVvVDrvFuncName = "llvm.riscv.vv.v.drv";
    
    func::FuncOp vvVDrvFunc = module.lookupSymbol<func::FuncOp>(kVvVDrvFuncName);
    if (!vvVDrvFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      vvVDrvFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), kVvVDrvFuncName, vvVDrv1FnType);
      vvVDrvFunc.setPrivate();
    }

    auto vvVDrvCall1 = rewriter.create<func::CallOp>(loc, vvVDrvFunc, vvVDrv1Operands);
    Value windowScalarUint32 = vvVDrvCall1.getResult(0);

    SmallVector<Value, 1> vSDrvOperands = {windowScalarUint32};
    SmallVector<Type, 1> vSDrvInputTypes = {uint32Type};
    auto vSDrvFnType = FunctionType::get(ctx, vSDrvInputTypes, {uint32Type});
    constexpr const char *kVSDrvFuncName = "llvm.riscv.v.s.drv";
    
    func::FuncOp vSDrvFunc = module.lookupSymbol<func::FuncOp>(kVSDrvFuncName);
    if (!vSDrvFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      vSDrvFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), kVSDrvFuncName, vSDrvFnType);
      vSDrvFunc.setPrivate();
    }

    auto vSDrvCall = rewriter.create<func::CallOp>(loc, vSDrvFunc, vSDrvOperands);
    Value maxScalarUint32 = vSDrvCall.getResult(0);

    SmallVector<Value, 2> vvVDrv2Operands = {maxScalarUint32, outputUint32};
    auto vvVDrvCall2 = rewriter.create<func::CallOp>(loc, vvVDrvFunc, vvVDrv2Operands);
    Value resultUint32 = vvVDrvCall2.getResult(0);

    if (output) {
      Value resultMemRef = buildValidMemRefFromUint32(resultUint32, output.getType(), rewriter, loc);
      rewriter.create<memref::CopyOp>(loc, resultMemRef, output);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: PoolingNCHWFchwOp operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ToLLVMFunc RewritePatterns: PoolingNCHWSumOp operations
//===----------------------------------------------------------------------===//

namespace {
class RAIRToCIMLoweringPass
    : public mlir::PassWrapper<RAIRToCIMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const final { 
    return "convert-rair-to-cim"; 
  }
  StringRef getDescription() const final {
    return "Lower RAIR dialect operations to sCIM dialect";
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RAIRToCIMLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<cim::CIMDialect, mlir::LLVM::LLVMDialect, 
                    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                    mlir::async::AsyncDialect, mlir::memref::MemRefDialect,
                    mlir::func::FuncDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void RAIRToCIMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());

  // target.addLegalDialect<cim::CIMDialect>();

  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                         mlir::async::AsyncDialect>();

  target.addIllegalOp<linalg::MatmulOp>();
  target.addIllegalDialect<rair::RAIRDialect, cim::CIMDialect>();

  target.addDynamicallyLegalDialect<mlir::linalg::LinalgDialect>(
    [](Operation *op) {
      // These specific ops are illegal, all others are legal
      return !isa<linalg::MatmulOp>(op);
    });
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<MatMulOpLoweringToCIM, BatchMatMulOpLoweringToCIM, 
   TransposeOpLoweringToCIM, 
   LoadOpLowering, StoreOpLowering,
   LMEMAllocOpConversion,LMEMDeallocOpConversion,
   WaitAllOpLowering,
   PoolingNCHWMaxOpLoweringToCIM>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace cim {  
  std::unique_ptr<mlir::Pass> createRAIRLowerToCIMPass() {
    return std::make_unique<RAIRToCIMLoweringPass>(); 
  }
}