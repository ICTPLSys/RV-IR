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

#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"
#include "torch-mlir/Dialect/RAIR/IR/RAIRDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>
#include <optional>

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/RAIR/IR/RAIROps.cpp.inc"

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

static bool isCoreMemorySpace(rair::MemorySpace space) {
  return space == rair::MemorySpace::HOST ||
         space == rair::MemorySpace::DEVICE ||
         space == rair::MemorySpace::SPAD ||
         space == rair::MemorySpace::ACC;
}

static rair::MemorySpaceAttr getTypedMemorySpace(MemRefType type) {
  return dyn_cast_or_null<rair::MemorySpaceAttr>(type.getMemorySpace());
}

static Operation *getAncestorInBlock(Operation *operation, Block *block) {
  while (operation && operation->getBlock() != block)
    operation = operation->getParentOp();
  return operation;
}

static FailureOr<int64_t> getStaticElementCount(rair::ViewOp view) {
  int64_t count = 1;
  for (int64_t size : view.getSizes()) {
    if (size <= 0 || count > std::numeric_limits<int64_t>::max() / size)
      return failure();
    count *= size;
  }
  return count;
}

} // namespace

namespace rair {

LogicalResult TargetOp::verify() {
  if (getKind().empty())
    return emitOpError("requires a non-empty target kind");
  if (auto bytes = getSpadBytesAttr(); bytes && bytes.getInt() <= 0)
    return emitOpError("requires spad_bytes to be positive");
  if (auto bytes = getAccBytesAttr(); bytes && bytes.getInt() <= 0)
    return emitOpError("requires acc_bytes to be positive");
  return success();
}

LogicalResult ScopeOp::verify() {
  auto target = SymbolTable::lookupNearestSymbolFrom<TargetOp>(
      getOperation(), getTargetAttr());
  if (!target)
    return emitOpError() << "references unknown rair.target " << getTarget();

  unsigned computeCount = 0;
  unsigned planCount = 0;
  bool sawPlan = false;
  for (Operation &operation : getBody().front()) {
    if (isa<ScopeTerminatorOp>(operation))
      continue;
    if (isa<PlanOp>(operation)) {
      sawPlan = true;
      if (++planCount > 1)
        return emitOpError("allows at most one associated rair.plan");
      continue;
    }
    if (sawPlan)
      return emitOpError(
          "requires its associated rair.plan after every Core operation");
    if (isa<ComputeOp>(operation))
      ++computeCount;
    if (!isa<ViewOp, ReserveOp, ReleaseLeaseOp, MoveOp, ComputeOp>(operation))
      return emitOpError()
             << "contains non-Core operation " << operation.getName()
             << " directly in its reference trace";
  }
  if (computeCount > 1)
    return emitOpError()
           << "supports at most one rair.compute in Core v0.1, but found "
           << computeCount;
  return success();
}

LogicalResult ViewOp::verify() {
  auto baseType = dyn_cast<MemRefType>(getBase().getType());
  if (!baseType || !baseType.hasStaticShape())
    return emitOpError("requires a ranked static-shape base memref");

  auto space = getTypedMemorySpace(baseType);
  if (!space || !isCoreMemorySpace(space.getValue()))
    return emitOpError(
        "requires a target-neutral #rair.space<host|device|spad|acc>");

  int64_t rank = baseType.getRank();
  if (static_cast<int64_t>(getOffsets().size()) != rank ||
      static_cast<int64_t>(getSizes().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("requires one offset, size, and stride per base rank");

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t offset = getOffsets()[dim];
    int64_t size = getSizes()[dim];
    int64_t stride = getStrides()[dim];
    int64_t bound = baseType.getDimSize(dim);
    if (offset < 0)
      return emitOpError() << "requires a non-negative offset at dimension "
                           << dim;
    if (size <= 0)
      return emitOpError() << "requires a positive size at dimension " << dim;
    if (stride <= 0)
      return emitOpError() << "requires a positive stride at dimension "
                           << dim;
    if (offset >= bound || size - 1 > (bound - 1 - offset) / stride)
      return emitOpError() << "region is out of bounds at dimension " << dim;
  }
  return success();
}

LogicalResult ReserveOp::verify() {
  auto bufferType = dyn_cast<MemRefType>(getBuffer().getType());
  if (!bufferType || !bufferType.hasStaticShape())
    return emitOpError("requires a ranked static-shape buffer result");
  if (!isCoreMemorySpace(getSpace()) || getSpace() == MemorySpace::HOST ||
      getSpace() == MemorySpace::UNKNOWN)
    return emitOpError(
        "requires #rair.space<device|spad|acc> for a reserved buffer");

  auto typeSpace = getTypedMemorySpace(bufferType);
  if (!typeSpace || typeSpace != getSpaceAttr())
    return emitOpError()
           << "requires result memref space to match " << getSpaceAttr();

  if (!getLease().hasOneUse())
    return emitOpError("requires its lease to have exactly one use");
  auto release = dyn_cast<ReleaseLeaseOp>(*getLease().getUsers().begin());
  if (!release || release.getBuffer() != getBuffer())
    return emitOpError(
        "requires its lease and buffer to be consumed by one matching "
        "rair.release_lease");
  if (release->getBlock() != getOperation()->getBlock() ||
      !getOperation()->isBeforeInBlock(release))
    return emitOpError(
        "requires rair.release_lease later in the same scope block");

  Block *scopeBlock = getOperation()->getBlock();
  auto verifyUseBeforeRelease = [&](Operation *owner,
                                    StringRef description) -> LogicalResult {
    Operation *anchor = getAncestorInBlock(owner, scopeBlock);
    if (!anchor)
      return emitOpError() << description << " escapes its rair.scope";
    if (anchor != release.getOperation() &&
        release->isBeforeInBlock(anchor))
      return emitOpError() << description << " is used after lease release";
    return success();
  };

  for (OpOperand &use : getBuffer().getUses()) {
    if (failed(verifyUseBeforeRelease(use.getOwner(), "reserved buffer")))
      return failure();
    if (auto view = dyn_cast<ViewOp>(use.getOwner())) {
      for (OpOperand &regionUse : view.getRegion().getUses())
        if (failed(verifyUseBeforeRelease(regionUse.getOwner(),
                                          "derived region")))
          return failure();
    }
  }
  return success();
}

LogicalResult ReleaseLeaseOp::verify() {
  auto reserve = getLease().getDefiningOp<ReserveOp>();
  if (!reserve)
    return emitOpError("requires a lease produced by rair.reserve");
  if (getBuffer() != reserve.getBuffer())
    return emitOpError("buffer does not match the lease's reservation");
  return success();
}

LogicalResult MoveOp::verify() {
  auto srcView = getSrc().getDefiningOp<ViewOp>();
  auto dstView = getDst().getDefiningOp<ViewOp>();
  if (!srcView || !dstView)
    return emitOpError("requires source and destination from rair.view");

  auto srcType = cast<MemRefType>(srcView.getBase().getType());
  auto dstType = cast<MemRefType>(dstView.getBase().getType());
  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError() << "does not perform implicit element conversion: "
                         << srcType.getElementType() << " versus "
                         << dstType.getElementType();

  FailureOr<int64_t> srcCount = getStaticElementCount(srcView);
  FailureOr<int64_t> dstCount = getStaticElementCount(dstView);
  if (failed(srcCount) || failed(dstCount) || *srcCount != *dstCount)
    return emitOpError("requires equal positive static element counts");
  return success();
}

LogicalResult ComputeOp::verify() {
  Operation *payload = nullptr;
  for (Operation &nested : getBody().front()) {
    if (isa<ComputeTerminatorOp>(nested))
      continue;
    if (payload)
      return emitOpError("expects exactly one payload operation");
    payload = &nested;
  }

  auto matmul = dyn_cast_or_null<linalg::MatmulOp>(payload);
  if (!matmul)
    return emitOpError(
        "expects one static memref linalg.matmul payload in Core v0.1");

  auto lhsView = getLhs().getDefiningOp<ViewOp>();
  auto rhsView = getRhs().getDefiningOp<ViewOp>();
  auto outputView = getOutput().getDefiningOp<ViewOp>();
  if (!lhsView || !rhsView || !outputView)
    return emitOpError("requires all footprint operands from rair.view");

  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      matmul.getInputs()[0] != lhsView.getBase() ||
      matmul.getInputs()[1] != rhsView.getBase() ||
      matmul.getOutputs()[0] != outputView.getBase())
    return emitOpError(
        "footprint regions must correspond to the matmul lhs, rhs, and output");

  auto lhsType = dyn_cast<MemRefType>(lhsView.getBase().getType());
  auto rhsType = dyn_cast<MemRefType>(rhsView.getBase().getType());
  auto outputType = dyn_cast<MemRefType>(outputView.getBase().getType());
  if (!lhsType || !rhsType || !outputType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outputType.hasStaticShape())
    return emitOpError("requires static memref matmul operands");

  auto lhsSpace = getTypedMemorySpace(lhsType);
  auto rhsSpace = getTypedMemorySpace(rhsType);
  auto outputSpace = getTypedMemorySpace(outputType);
  if (!lhsSpace || !rhsSpace || !outputSpace ||
      lhsSpace.getValue() != MemorySpace::SPAD ||
      rhsSpace.getValue() != MemorySpace::SPAD ||
      outputSpace.getValue() != MemorySpace::ACC)
    return emitOpError(
        "requires matmul inputs in #rair.space<spad> and output in "
        "#rair.space<acc>");
  return success();
}

LogicalResult PlanOp::verify() {
  auto scope = dyn_cast_or_null<ScopeOp>(getOperation()->getParentOp());
  if (!scope)
    return emitOpError("must be directly nested in rair.scope");
  if (getTargetAttr() != scope.getTargetAttr())
    return emitOpError()
           << "targets " << getTargetAttr() << " but containing scope targets "
           << scope.getTargetAttr();

  auto target = SymbolTable::lookupNearestSymbolFrom<TargetOp>(
      getOperation(), getTargetAttr());
  if (!target)
    return emitOpError()
           << "references unknown rair.target " << getTargetAttr();

  llvm::DenseSet<int64_t> sourceActionIds;
  int64_t taskCount = 0;
  for (Operation &operation : getBody().front()) {
    if (isa<PlanTerminatorOp>(operation))
      continue;

    auto task = dyn_cast<TaskOp>(operation);
    if (!task)
      return emitOpError()
             << "contains non-Plan operation " << operation.getName()
             << " directly in its task graph";

    ++taskCount;
    int64_t sourceAction = task.getSourceAction();
    if (sourceAction < 0)
      return task.emitOpError("requires a non-negative source_action ID");
    if (!sourceActionIds.insert(sourceAction).second)
      return task.emitOpError()
             << "duplicates source_action ID " << sourceAction
             << " in the containing rair.plan";
  }

  for (int64_t sourceAction = 0; sourceAction < taskCount; ++sourceAction)
    if (!sourceActionIds.contains(sourceAction))
      return emitOpError()
             << "requires dense source_action IDs [0, " << taskCount
             << "), but ID " << sourceAction << " is missing";

  return success();
}

LogicalResult TaskOp::verify() {
  auto plan = dyn_cast_or_null<PlanOp>(getOperation()->getParentOp());
  if (!plan)
    return emitOpError("must be directly nested in rair.plan");

  auto eventType = cast<EventType>(getEvent().getType());
  if (eventType.getTarget() != plan.getTargetAttr())
    return emitOpError()
           << "produces event for " << eventType.getTarget()
           << " but containing plan targets " << plan.getTargetAttr();

  llvm::DenseSet<Value> seenDependencies;
  for (Value dependency : getDependencies()) {
    auto dependencyType = cast<EventType>(dependency.getType());
    if (dependencyType.getTarget() != plan.getTargetAttr())
      return emitOpError()
             << "has dependency event for " << dependencyType.getTarget()
             << " but containing plan targets " << plan.getTargetAttr();
    if (!seenDependencies.insert(dependency).second)
      return emitOpError("lists the same dependency event more than once");

    auto producer = dependency.getDefiningOp<TaskOp>();
    if (!producer || producer->getParentOp() != plan.getOperation())
      return emitOpError(
          "requires every dependency event to be produced by a task in the "
          "same rair.plan");
    if (producer->getBlock() != getOperation()->getBlock() ||
        !producer->isBeforeInBlock(getOperation()))
      return emitOpError(
          "requires every dependency event producer to precede its consumer");
  }

  return success();
}

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
