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

#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.h"
#include "torch-mlir/Dialect/RISCV/IR/RISCVDialect.h"
#include "mlir/IR/OpImplementation.h"

#include <optional>

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/RISCV/IR/RISCVOps.cpp.inc"

using namespace mlir;

namespace {

static bool areCompatibleDims(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) ||
         lhs == rhs;
}

static Attribute getMemRefMemorySpace(Type type) {
  return cast<BaseMemRefType>(type).getMemorySpace();
}

static std::optional<rair::MemorySpace>
decodeMemorySpace(Attribute memorySpace) {
  if (auto typedSpace = dyn_cast<rair::MemorySpaceAttr>(memorySpace))
    return typedSpace.getValue();

  auto integerSpace = dyn_cast<IntegerAttr>(memorySpace);
  if (!integerSpace)
    return std::nullopt;

  int64_t value = integerSpace.getInt();
  if (value < 0 ||
      value > static_cast<int64_t>(rair::getMaxEnumValForMemorySpace()))
    return std::nullopt;
  return rair::symbolizeMemorySpace(static_cast<uint32_t>(value));
}

static LogicalResult verifyMemorySpaceConsistency(
    Operation *op, StringRef attributeName,
    rair::MemorySpaceAttr attributeSpace, Type memrefType,
    StringRef valueDescription) {
  if (!attributeSpace)
    return success();

  Attribute typeSpace = getMemRefMemorySpace(memrefType);
  if (!typeSpace)
    return success();

  std::optional<rair::MemorySpace> decodedTypeSpace =
      decodeMemorySpace(typeSpace);
  if (!decodedTypeSpace)
    return op->emitOpError()
           << "has unsupported " << valueDescription
           << " memref memory space " << typeSpace;

  if (*decodedTypeSpace != attributeSpace.getValue())
    return op->emitOpError()
           << "has inconsistent " << attributeName << ": attribute is "
           << attributeSpace << " but " << valueDescription
           << " memref type uses " << typeSpace;

  return success();
}

} // namespace

namespace rair {

LogicalResult MatmulOp::verify() {
  auto lhsType = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(getRhs().getType());
  auto outputType = dyn_cast<MemRefType>(getOutput().getType());

  if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outputType.getRank() != 2)
    return emitOpError("expects lhs, rhs, and output to be rank-2 memrefs");

  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType())
    return emitOpError()
           << "expects matching element types, but got lhs "
           << lhsType.getElementType() << ", rhs " << rhsType.getElementType()
           << ", and output " << outputType.getElementType();

  if (!areCompatibleDims(lhsType.getDimSize(1), rhsType.getDimSize(0)))
    return emitOpError()
           << "has incompatible contracting dimensions: lhs dimension 1 is "
           << lhsType.getDimSize(1) << " but rhs dimension 0 is "
           << rhsType.getDimSize(0);

  if (!areCompatibleDims(lhsType.getDimSize(0), outputType.getDimSize(0)))
    return emitOpError()
           << "has incompatible row dimensions: lhs dimension 0 is "
           << lhsType.getDimSize(0) << " but output dimension 0 is "
           << outputType.getDimSize(0);

  if (!areCompatibleDims(rhsType.getDimSize(1), outputType.getDimSize(1)))
    return emitOpError()
           << "has incompatible column dimensions: rhs dimension 1 is "
           << rhsType.getDimSize(1) << " but output dimension 1 is "
           << outputType.getDimSize(1);

  return success();
}

LogicalResult BatchMatMulOp::verify() {
  auto lhsType = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(getRhs().getType());
  auto outputType = dyn_cast<MemRefType>(getOutput().getType());

  if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 3 ||
      rhsType.getRank() != 3 || outputType.getRank() != 3)
    return emitOpError("expects lhs, rhs, and output to be rank-3 memrefs");

  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType())
    return emitOpError()
           << "expects matching element types, but got lhs "
           << lhsType.getElementType() << ", rhs " << rhsType.getElementType()
           << ", and output " << outputType.getElementType();

  if (!areCompatibleDims(lhsType.getDimSize(0), rhsType.getDimSize(0)) ||
      !areCompatibleDims(lhsType.getDimSize(0), outputType.getDimSize(0)) ||
      !areCompatibleDims(rhsType.getDimSize(0), outputType.getDimSize(0)))
    return emitOpError()
           << "has incompatible batch dimensions: lhs dimension 0 is "
           << lhsType.getDimSize(0) << ", rhs dimension 0 is "
           << rhsType.getDimSize(0) << ", and output dimension 0 is "
           << outputType.getDimSize(0);

  if (!areCompatibleDims(lhsType.getDimSize(2), rhsType.getDimSize(1)))
    return emitOpError()
           << "has incompatible contracting dimensions: lhs dimension 2 is "
           << lhsType.getDimSize(2) << " but rhs dimension 1 is "
           << rhsType.getDimSize(1);

  if (!areCompatibleDims(lhsType.getDimSize(1), outputType.getDimSize(1)))
    return emitOpError()
           << "has incompatible row dimensions: lhs dimension 1 is "
           << lhsType.getDimSize(1) << " but output dimension 1 is "
           << outputType.getDimSize(1);

  if (!areCompatibleDims(rhsType.getDimSize(2), outputType.getDimSize(2)))
    return emitOpError()
           << "has incompatible column dimensions: rhs dimension 2 is "
           << rhsType.getDimSize(2) << " but output dimension 2 is "
           << outputType.getDimSize(2);

  return success();
}

LogicalResult TransferOp::verify() {
  auto srcType = cast<ShapedType>(getSrc().getType());
  auto dstType = cast<ShapedType>(getDst().getType());

  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError() << "expects matching element types, but got source "
                         << srcType.getElementType() << " and destination "
                         << dstType.getElementType();

  if (failed(verifyMemorySpaceConsistency(
          getOperation(), "src_memory_space", getSrcMemorySpaceAttr(),
          getSrc().getType(), "source")))
    return failure();
  if (failed(verifyMemorySpaceConsistency(
          getOperation(), "dst_memory_space", getDstMemorySpaceAttr(),
          getDst().getType(), "destination")))
    return failure();

  auto rankedSrcType = dyn_cast<MemRefType>(getSrc().getType());
  auto rankedDstType = dyn_cast<MemRefType>(getDst().getType());
  if (!rankedSrcType || !rankedDstType)
    return success();

  if (rankedSrcType.getRank() != rankedDstType.getRank())
    return emitOpError()
           << "expects source and destination to have the same rank, but got "
           << rankedSrcType.getRank() << " and " << rankedDstType.getRank();

  for (int64_t dim = 0; dim < rankedSrcType.getRank(); ++dim) {
    if (!areCompatibleDims(rankedSrcType.getDimSize(dim),
                           rankedDstType.getDimSize(dim)))
      return emitOpError()
             << "has incompatible static size at dimension " << dim
             << ": source is " << rankedSrcType.getDimSize(dim)
             << " but destination is " << rankedDstType.getDimSize(dim);
  }

  return success();
}

LogicalResult AllocBufferOp::verify() {
  return verifyMemorySpaceConsistency(
      getOperation(), "memory_space", getMemorySpaceAttr(),
      getResult().getType(), "result");
}

} // namespace rair
