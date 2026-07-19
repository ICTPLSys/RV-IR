//===- VerifyLifetimes.cpp - Verify RAIR linear resource lifetimes --------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h"

#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static bool isLifetimeResourceOp(Operation *op) {
  return isa<rair::AcquireOp, rair::ReleaseOp, rair::AwaitOp,
             rair::AllocBufferOp, rair::DeallocBufferOp>(op);
}

static LogicalResult verifyResourceStructure(func::FuncOp function,
                                             Block *&entryBlock) {
  SmallVector<Operation *> resourceOps;
  function.walk([&](Operation *op) {
    if (isLifetimeResourceOp(op))
      resourceOps.push_back(op);
  });

  if (resourceOps.empty()) {
    entryBlock = nullptr;
    return success();
  }

  if (!llvm::hasSingleElement(function.getBody()))
    return function.emitError(
        "RAIR lifetime verification currently supports only single-block "
        "functions; resource lifetimes across CFG branches are unsupported");

  entryBlock = &function.getBody().front();
  for (Operation *op : resourceOps) {
    if (op->getBlock() != entryBlock)
      return op->emitOpError(
          "is nested in a region; RAIR lifetime verification does not yet "
          "support resources inside branches, loops, or nested regions");
  }
  return success();
}

static LogicalResult verifyReleaseOrigins(Block &block) {
  for (rair::ReleaseOp release : block.getOps<rair::ReleaseOp>()) {
    auto acquire = release.getContext().getDefiningOp<rair::AcquireOp>();
    if (!acquire || acquire->getBlock() != &block)
      return release.emitOpError(
          "must consume a context produced by 'rair.acquire' in the same "
          "straight-line function block");
  }
  return success();
}

static LogicalResult verifyDeallocOrigins(Block &block) {
  for (rair::DeallocBufferOp dealloc :
       block.getOps<rair::DeallocBufferOp>()) {
    auto alloc = dealloc.getBuffer().getDefiningOp<rair::AllocBufferOp>();
    if (!alloc || alloc->getBlock() != &block)
      return dealloc.emitOpError(
          "must consume a buffer produced by 'rair.alloc_buffer' in the same "
          "straight-line function block");
  }
  return success();
}

static LogicalResult verifyContextLifetime(rair::AcquireOp acquire,
                                           Block &block) {
  SmallVector<rair::ReleaseOp> releases;
  for (OpOperand &use : acquire.getContext().getUses()) {
    if (auto release = dyn_cast<rair::ReleaseOp>(use.getOwner()))
      releases.push_back(release);
  }

  for (OpOperand &use : acquire.getContext().getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getBlock() != &block)
      return owner->emitOpError(
          "uses an acquired RAIR context in a nested region; resource "
          "lifetimes across branches and loops are unsupported");
    if (isa<rair::ReleaseOp, rair::AllocBufferOp, rair::DeallocBufferOp,
            rair::AwaitOp>(owner))
      continue;
    if (owner->hasTrait<OpTrait::IsTerminator>())
      return owner->emitOpError(
          "returns or yields an acquired RAIR context; context escape is "
          "unsupported by 'rair-verify-lifetimes'");
    return owner->emitOpError(
        "may not consume an acquired RAIR context; passing owned contexts to "
        "calls or unknown operations is unsupported");
  }

  if (releases.size() != 1)
    return acquire.emitOpError()
           << "context must be released exactly once in the same "
              "straight-line function block, but found "
           << releases.size() << " 'rair.release' operations";

  Operation *release = releases.front().getOperation();
  for (OpOperand &use : acquire.getContext().getUses()) {
    Operation *owner = use.getOwner();
    if (owner != release && release->isBeforeInBlock(owner))
      return owner->emitOpError(
          "uses an acquired RAIR context after 'rair.release'");
  }
  return success();
}

static bool hasMemRefResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<BaseMemRefType>(type); });
}

static LogicalResult verifyBufferLifetime(rair::AllocBufferOp alloc,
                                          Block &block) {
  SmallVector<rair::DeallocBufferOp> deallocs;
  for (OpOperand &use : alloc.getResult().getUses()) {
    if (auto dealloc = dyn_cast<rair::DeallocBufferOp>(use.getOwner()))
      deallocs.push_back(dealloc);
  }

  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getBlock() != &block)
      return owner->emitOpError(
          "uses an owned RAIR buffer in a nested region; resource lifetimes "
          "across branches and loops are unsupported");
    if (isa<rair::DeallocBufferOp>(owner))
      continue;
    if (owner->hasTrait<OpTrait::IsTerminator>())
      return owner->emitOpError(
          "returns or yields an owned RAIR buffer; buffer escape is "
          "unsupported by 'rair-verify-lifetimes'");
    if (isa<func::CallOp, func::CallIndirectOp>(owner))
      return owner->emitOpError(
          "passes an owned RAIR buffer to a call; interprocedural ownership "
          "is unsupported by 'rair-verify-lifetimes'");
    if (owner->getNumRegions() != 0)
      return owner->emitOpError(
          "captures an owned RAIR buffer in a nested region; resource "
          "lifetimes across branches and loops are unsupported");
    if (hasMemRefResult(owner))
      return owner->emitOpError(
          "creates a derived memref from an owned RAIR buffer; alias-aware "
          "lifetime verification is not yet supported");
  }

  if (deallocs.size() != 1)
    return alloc.emitOpError()
           << "owned buffer must be deallocated exactly once in the same "
              "straight-line function block, but found "
           << deallocs.size() << " 'rair.dealloc_buffer' operations";

  rair::DeallocBufferOp dealloc = deallocs.front();
  if (dealloc.getContext() != alloc.getContext())
    return dealloc.emitOpError(
        "must use the same context as the corresponding "
        "'rair.alloc_buffer'");

  Operation *deallocOperation = dealloc.getOperation();
  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner != deallocOperation && deallocOperation->isBeforeInBlock(owner))
      return owner->emitOpError(
          "uses an owned RAIR buffer after 'rair.dealloc_buffer'");
  }
  return success();
}

static LogicalResult verifyFunctionLifetimes(func::FuncOp function) {
  Block *entryBlock = nullptr;
  if (failed(verifyResourceStructure(function, entryBlock)))
    return failure();
  if (!entryBlock)
    return success();

  if (failed(verifyReleaseOrigins(*entryBlock)) ||
      failed(verifyDeallocOrigins(*entryBlock)))
    return failure();

  for (rair::AcquireOp acquire : entryBlock->getOps<rair::AcquireOp>()) {
    if (failed(verifyContextLifetime(acquire, *entryBlock)))
      return failure();
  }
  for (rair::AllocBufferOp alloc :
       entryBlock->getOps<rair::AllocBufferOp>()) {
    if (failed(verifyBufferLifetime(alloc, *entryBlock)))
      return failure();
  }
  return success();
}

class VerifyLifetimesPass
    : public PassWrapper<VerifyLifetimesPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyLifetimesPass)

  StringRef getArgument() const final { return "rair-verify-lifetimes"; }
  StringRef getDescription() const final {
    return "Verify straight-line RAIR context and buffer lifetimes";
  }

  void runOnOperation() final {
    if (failed(verifyFunctionLifetimes(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
rair::createVerifyLifetimesPass() {
  return std::make_unique<VerifyLifetimesPass>();
}
