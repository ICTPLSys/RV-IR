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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

#include "torch-mlir/Dialect/RISCV/IR/RISCVDialect.h"
#include "torch-mlir/Dialect/RISCV/IR//RISCVOps.h"
// #include "RISCV/RISCVConstant.h"

using namespace mlir;
using namespace riscv;   

//===----------------------------------------------------------------------===//
// RISCV dialect.
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RISCV/IR/RISCVDialect.cpp.inc"

// #include "RISCV/RISCVDialect.cpp.inc"


void RISCVDialect::initialize() {
//     addTypes<
// #define GET_TYPEDEF_LIST
// #include "RISCV/RISCVOpsTypes.cpp.inc"
//       >();
  addTypes<AsyncTokenType>();
  addTypes<EventType>();
  addOperations<
#define GET_OP_LIST
#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.cpp.inc"
      >();

}

Type RISCVDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'event' types.
  if (keyword == "event")
    return EventType::get(context);

  // Handle 'async_token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown riscv type: " + keyword);
  return Type();
}

void RISCVDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<EventType>([&](Type) { os << "event"; })
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'riscv' type"); });
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                    llvm::StringRef name, mlir::FunctionType type,
//                    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
//   // FunctionOpInterface provides a convenient `build` method that will populate
//   // the state of our FuncOp, and create an entry block.
//   buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
// }
// mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
//                                 mlir::OperationState &result) {
//   // Dispatch to the FunctionOpInterface provided utility method that parses the
//   // function operation.
//   auto buildFuncType =
//       [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
//          llvm::ArrayRef<mlir::Type> results,
//          mlir::function_interface_impl::VariadicFlag,
//          std::string &) { return builder.getFunctionType(argTypes, results); };

//   return mlir::function_interface_impl::parseFunctionOp(
//       parser, result, /*allowVariadic=*/false,
//       getFunctionTypeAttrName(result.name), buildFuncType,
//       getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
// }

// void FuncOp::print(mlir::OpAsmPrinter &p) {
//   // Dispatch to the FunctionOpInterface provided utility method that prints the
//   // function operation.
//   mlir::function_interface_impl::printFunctionOp(
//       p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
//       getArgAttrsAttrName(), getResAttrsAttrName());
// }
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
// void ConstantOp::build(mlir::OpBuilder &builder,
//                               mlir::OperationState &state, double value) {
//   auto dataType = RankedTensorType::get({}, builder.getF64Type());
//   auto dataAttribute = DenseElementsAttr::get(dataType, value);
//   riscv::ConstantOp::build(builder, state, dataType, dataAttribute);
// }
// 构建一个 f64 类型的常量
void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
  auto type = RankedTensorType::get({}, builder.getF64Type());
  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef<double>(value));
  state.addAttribute("value", attr);
  state.addTypes(type);
}

void ConstantOp::build(OpBuilder &builder, OperationState &state, int64_t value) {
  auto type = RankedTensorType::get({}, builder.getI64Type());
  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef<int64_t>(value));
  state.addAttribute("value", attr);
  state.addTypes(type);
}
// void ConstantOp::build(mlir::OpBuilder &builder,
//                        mlir::OperationState &state, int64_t value) {
//   auto dataType = mlir::RankedTensorType::get({}, builder.getI64Type());
//   auto attr = builder.getI64IntegerAttr(value);
//   auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
//   riscv::ConstantOp::build(builder, state, dataType, dataAttribute);
// }
//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

// LogicalResult TransposeOp::inferReturnTypes(
//     MLIRContext *context, std::optional<Location> location,
//     TransposeOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
//   inferredReturnTypes.assign({IndexType::get(context)});
//   return success();
// }
// void TransposeOp::inferShapes() {
//   auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
//   SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
//   getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
// }

// mlir::LogicalResult TransposeOp::verify() {
//   auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
//   auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
//   if (!inputType || !resultType)
//     return mlir::success();

//   auto inputShape = inputType.getShape();
//   if (!std::equal(inputShape.begin(), inputShape.end(),
//                   resultType.getShape().rbegin())) {
//     return emitError()
//            << "expected result shape to be a transpose of the input";
//   }
//   return mlir::success();
// }
//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }
void AddOp::build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs) {
  // 假设输入类型一致
  auto resultType = lhs.getType();
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
}
//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

// void AddFOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }
//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

// void AddIOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }

// mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
//                                mlir::OperationState &result) {
//   return parseBinaryOp(parser, result);
// }

// void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
// void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

// void SubFOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

// void SubIOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

// void MulFOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }
//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

// void MulIOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

void MatmulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value lhs, mlir::Value rhs) {
  // Get the element type from the left operand
  auto lhsType = mlir::cast<TensorType>(lhs.getType());
  auto elementType = lhsType.getElementType();
  
  // Build result type with same element type
  state.addTypes(UnrankedTensorType::get(elementType));
  state.addOperands({lhs, rhs});
}

// mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
//                                mlir::OperationState &result) {
//   return parseBinaryOp(parser, result);
// }

// void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
// void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// MatvecOp
//===----------------------------------------------------------------------===//

void MatvecOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value lhs, mlir::Value rhs) {
  // Get the element type from the left operand
  auto lhsType = mlir::cast<TensorType>(lhs.getType());
  auto elementType = lhsType.getElementType();
  
  // Build result type with same element type
  state.addTypes(UnrankedTensorType::get(elementType));
  state.addOperands({lhs, rhs});
}
//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//
void ReduceOp::build(OpBuilder &builder, OperationState &state,Type resultType,
                     Value input, llvm::ArrayRef<int64_t> dim,
                     llvm::StringRef kind) {
  state.addOperands(input);
  auto dimAttr = builder.getI64ArrayAttr(dim);
  auto kindAttr = builder.getStringAttr(kind);
  state.addAttribute("dim", dimAttr);
  state.addAttribute("kind", kindAttr);
  // auto resultType = input.getType();
  state.addTypes(resultType);
}
//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//
// void Conv2DOp::build(OpBuilder &builder, OperationState &state,
//                      Value input, Value kernel, Value bias) {
//   state.addOperands({input, kernel, bias});
//   auto resultType = input.getType();
//   state.addTypes(resultType);
// }
void Conv2DOp::build(OpBuilder &builder, OperationState &state,
                     Value input, Value kernel) {
  state.addOperands({input, kernel});
  auto resultType = input.getType();
  state.addTypes(resultType);
}
//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

// void DivFOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }
//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

// void DivSIOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }
//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

// void DivUIOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
//   state.addOperands({lhs, rhs});
// }

//===----------------------------------------------------------------------===//
// FutureType
//===----------------------------------------------------------------------===//
// Type FutureType::parse(mlir::AsmParser &parser) {
//   SmallVector<int64_t> shape;
//   Type elementType;

//   if (parser.parseLess() ||                            //
//       parser.parseDimensionList(shape, false, true) || //
//       parser.parseType(elementType) ||                 //
//       parser.parseGreater()) {
//     return Type();
//   }

//   return FutureType::get(parser.getContext(), shape, elementType);
// }

// void FutureType::print(mlir::AsmPrinter &printer) const {
//   printer << "<";
//   printer.printDimensionList(getShape());
//   printer << (getShape().empty() ? "" : "x") << getElementType() << ">";
// }


//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//
void riscv::LaunchOp::print(OpAsmPrinter &p) {

  p << ' ';

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this,
                         (getAsyncToken() ? getAsyncToken().getType() : Type()),
                         getAsyncDependencies());
  p << "(";
  p.printOperands(getIds());
  p << ") in (";
  auto sizeArgs = getSize();
  auto sizeOpers = getSizeOperands();
  for (int i = 0, e = getNumDims(); i < e; i++) {
    if (i)
      p << ", ";
    p << sizeArgs[i] << "=";
    p << sizeOpers[i];
  }
  p << ")";

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
        if (attr.getName() == getOperandSegmentSizeAttr())
          return false;
        if (attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          return false;
        return true;
      }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  if (nameAttr && getBody().front().getOperations().size() == 1)
    return;
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult riscv::LaunchOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  if (parser.parseArgumentList(tileArgs, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();

  tileSize.resize(tileArgs.size());
  tileSizeRef.resize(tileArgs.size());
  for (unsigned i = 0; i < tileArgs.size(); ++i) {
    if (parser.parseArgument(tileSizeRef[i]) || parser.parseEqual() ||
        parser.parseOperand(tileSize[i]))
      return failure();
    (void)parser.parseOptionalComma();
  }

  if (parser.parseRParen())
    return failure();

  Type indexType = parser.getBuilder().getIndexType();

  tileArgs.append(tileSizeRef);
  for (auto &a : tileArgs)
    a.type = indexType;

  auto tokenType = AsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependencies, tokenType, result.operands))
    return failure();
  if (parser.resolveOperands(tileSize, indexType, result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::Argument argument;
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseArgument(argument) || parser.parseEqual() ||
            parser.parseOperand(operand))
          return failure();
        kernelArguments.push_back(argument);
        kernelOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
      if (parser.parseColonTypeList(types))
        return failure();
    }
  }

  for (int i = 0, e = kernelOperands.size(); i < e; i++) {
    kernelArguments[i].type = types[i];
    tileArgs.push_back(kernelArguments[i]);
    if (parser.resolveOperand(kernelOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();

  auto regionResult = parser.parseOptionalRegion(*body, tileArgs);
  LaunchOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  if (!regionResult.has_value()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}
ArrayRef<BlockArgument> riscv::LaunchOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> riscv::LaunchOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange riscv::LaunchOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned riscv::LaunchOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

OperandRange riscv::LaunchOp::getKernelOperands() {
  return getOperands().drop_front(getAsyncDependencies().size() + getNumDims());
}

Value riscv::LaunchOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> riscv::LaunchOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument riscv::LaunchOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned riscv::LaunchOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}
//===----------------------------------------------------------------------===//
// HerdOp
//===----------------------------------------------------------------------===//
ArrayRef<BlockArgument> riscv::HerdOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> riscv::HerdOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange riscv::HerdOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned riscv::HerdOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

OperandRange riscv::HerdOp::getKernelOperands() {
  return getOperands().drop_front(getAsyncDependencies().size() + getNumDims());
}

Value riscv::HerdOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> riscv::HerdOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(4);
}

BlockArgument riscv::HerdOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned riscv::HerdOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

uint64_t riscv::HerdOp::getNumCols() {
  auto cols = getSizeOperands()[0].getDefiningOp();
  return cast<arith::ConstantIndexOp>(cols).value();
}

uint64_t riscv::HerdOp::getNumRows() {
  auto rows = getSizeOperands()[1].getDefiningOp();
  return cast<arith::ConstantIndexOp>(rows).value();
}

mlir::Operation *RISCVDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<riscv::ConstantOp>(
      loc, type, mlir::cast<mlir::DenseElementsAttr>(value));
}