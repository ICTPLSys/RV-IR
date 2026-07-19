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

#ifndef TORCHMLIR_DIALECT_RAIR_IR_RAIRDIALECT_H
#define TORCHMLIR_DIALECT_RAIR_IR_RAIRDIALECT_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "torch-mlir/Dialect/RAIR/IR/RAIRDialect.h.inc"
#include "torch-mlir/Dialect/RAIR/IR/RAIREnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "torch-mlir/Dialect/RAIR/IR/RAIRTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "torch-mlir/Dialect/RAIR/IR/RAIRAttributes.h.inc"

using namespace mlir;
//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//
namespace rair{
static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<rair::AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependenciesUnsorted) {

  if (asyncTokenType)
    printer << "async ";
  if (asyncDependenciesUnsorted.empty())
    return;

  // The values can be sorted by their order in a basic block, but only if they
  // all have defining ops in the same basic block. We go through all the
  // values, and check that they have defining ops in the same block.
  bool canSort = [&]() {
    auto v0 = asyncDependenciesUnsorted[0];
    if (!v0.getDefiningOp())
      return false;
    auto block = v0.getDefiningOp()->getBlock();
    for (auto v : asyncDependenciesUnsorted) {
      auto op = v.getDefiningOp();
      if (!op)
        return false;
      auto b = op->getBlock();
      if (b != block)
        return false;
    }
    return true;
  }();

  printer << "[";

  if (!canSort) {
    llvm::interleaveComma(asyncDependenciesUnsorted, printer);
  } else {
    SmallVector<Value> asyncDependencies(asyncDependenciesUnsorted);
    llvm::sort(asyncDependencies, [&](Value a, Value b) {
      return a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
    });
    llvm::interleaveComma(asyncDependencies, printer);
  }
  printer << "] ";
}
}

#endif // TORCHMLIR_DIALECT_RAIR_IR_RAIRDIALECT_H

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: RAIRDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// namespace rair {

// class RAIRDialect : public ::mlir::Dialect {
//   explicit RAIRDialect(::mlir::MLIRContext *context);

//   void initialize();
//   friend class ::mlir::MLIRContext;
// public:
//   ~RAIRDialect() override;
//   static constexpr ::llvm::StringLiteral getDialectNamespace() {
//     return ::llvm::StringLiteral("rair");
//   }

//   /// Materialize a single constant operation from a given attribute value with
//   /// the desired resultant type.
//   ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
//                                          ::mlir::Attribute value,
//                                          ::mlir::Type type,
//                                          ::mlir::Location loc) override;
// };
// } // namespace rair
// MLIR_DECLARE_EXPLICIT_TYPE_ID(::rair::RAIRDialect)
