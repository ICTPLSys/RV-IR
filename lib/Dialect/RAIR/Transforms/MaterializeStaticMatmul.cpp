//===- MaterializeStaticMatmul.cpp - Build a RAIR Core matmul trace -------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h"

#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kDefaultTargetName = "rair_default";
constexpr llvm::StringLiteral kDefaultTargetKind = "generic";

static std::optional<rair::MemorySpace> getTypedSpace(MemRefType type) {
  auto space = dyn_cast_or_null<rair::MemorySpaceAttr>(type.getMemorySpace());
  if (!space)
    return std::nullopt;
  return space.getValue();
}

static bool isExternalCoreSpace(rair::MemorySpace space) {
  return space == rair::MemorySpace::HOST || space == rair::MemorySpace::DEVICE;
}

static LogicalResult validateMatmul(linalg::MatmulOp matmul) {
  if (matmul->getNumResults() != 0)
    return matmul.emitOpError(
        "requires buffer semantics; tensor-result matmul is not supported");
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1)
    return matmul.emitOpError("requires exactly two inputs and one output");

  SmallVector<Value> operands = {matmul.getInputs()[0], matmul.getInputs()[1],
                                 matmul.getOutputs()[0]};
  for (auto [index, operand] : llvm::enumerate(operands)) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape())
      return matmul.emitOpError()
             << "requires rank-2 static memref operand " << index;

    std::optional<rair::MemorySpace> space = getTypedSpace(type);
    if (!space || !isExternalCoreSpace(*space))
      return matmul.emitOpError()
             << "requires operand " << index
             << " in #rair.space<host> or #rair.space<device>";
  }
  return success();
}

static MemRefType makeLocalType(MemRefType externalType,
                                rair::MemorySpace space) {
  auto memorySpace =
      rair::MemorySpaceAttr::get(externalType.getContext(), space);
  return MemRefType::get(externalType.getShape(), externalType.getElementType(),
                         MemRefLayoutAttrInterface(), memorySpace);
}

static rair::ViewOp createWholeView(OpBuilder &builder, Location location,
                                    Value base) {
  auto type = cast<MemRefType>(base.getType());
  SmallVector<int64_t> offsets(type.getRank(), 0);
  SmallVector<int64_t> strides(type.getRank(), 1);
  return builder.create<rair::ViewOp>(
      location, rair::RegionType::get(builder.getContext()), base, offsets,
      type.getShape(), strides);
}

static void materializeMatmul(linalg::MatmulOp matmul) {
  Location location = matmul.getLoc();
  MLIRContext *context = matmul.getContext();
  Value lhs = matmul.getInputs()[0];
  Value rhs = matmul.getInputs()[1];
  Value output = matmul.getOutputs()[0];
  auto lhsType = cast<MemRefType>(lhs.getType());
  auto rhsType = cast<MemRefType>(rhs.getType());
  auto outputType = cast<MemRefType>(output.getType());

  OpBuilder builder(matmul);
  auto scope =
      builder.create<rair::ScopeOp>(location, kDefaultTargetName.str());
  builder.createBlock(&scope.getBody());
  builder.setInsertionPointToEnd(&scope.getBody().front());

  rair::ViewOp lhsExternal = createWholeView(builder, location, lhs);
  rair::ViewOp rhsExternal = createWholeView(builder, location, rhs);
  rair::ViewOp outputExternal = createWholeView(builder, location, output);

  auto leaseType = rair::LeaseType::get(context);
  auto lhsReserve = builder.create<rair::ReserveOp>(
      location, makeLocalType(lhsType, rair::MemorySpace::SPAD), leaseType,
      rair::MemorySpace::SPAD);
  auto rhsReserve = builder.create<rair::ReserveOp>(
      location, makeLocalType(rhsType, rair::MemorySpace::SPAD), leaseType,
      rair::MemorySpace::SPAD);
  auto outputReserve = builder.create<rair::ReserveOp>(
      location, makeLocalType(outputType, rair::MemorySpace::ACC), leaseType,
      rair::MemorySpace::ACC);

  rair::ViewOp lhsLocal =
      createWholeView(builder, location, lhsReserve.getBuffer());
  rair::ViewOp rhsLocal =
      createWholeView(builder, location, rhsReserve.getBuffer());
  rair::ViewOp outputLocal =
      createWholeView(builder, location, outputReserve.getBuffer());

  builder.create<rair::MoveOp>(location, lhsExternal.getRegion(),
                               lhsLocal.getRegion());
  builder.create<rair::MoveOp>(location, rhsExternal.getRegion(),
                               rhsLocal.getRegion());
  // linalg.matmul reads and accumulates into its output. Preserve the initial
  // output value instead of assuming that a fresh ACC reservation is zeroed.
  builder.create<rair::MoveOp>(location, outputExternal.getRegion(),
                               outputLocal.getRegion());

  auto compute = builder.create<rair::ComputeOp>(location, lhsLocal.getRegion(),
                                                 rhsLocal.getRegion(),
                                                 outputLocal.getRegion());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&compute.getBody());
    builder.setInsertionPointToEnd(&compute.getBody().front());
    Operation *payload = builder.clone(*matmul.getOperation());
    SmallVector<Value> localOperands = {lhsReserve.getBuffer(),
                                        rhsReserve.getBuffer(),
                                        outputReserve.getBuffer()};
    payload->setOperands(localOperands);
    builder.create<rair::ComputeTerminatorOp>(location);
  }

  builder.create<rair::MoveOp>(location, outputLocal.getRegion(),
                               outputExternal.getRegion());
  builder.create<rair::ReleaseLeaseOp>(location, outputReserve.getLease(),
                                       outputReserve.getBuffer());
  builder.create<rair::ReleaseLeaseOp>(location, rhsReserve.getLease(),
                                       rhsReserve.getBuffer());
  builder.create<rair::ReleaseLeaseOp>(location, lhsReserve.getLease(),
                                       lhsReserve.getBuffer());
  builder.create<rair::ScopeTerminatorOp>(location);

  matmul.erase();
}

class MaterializeStaticMatmulPass
    : public PassWrapper<MaterializeStaticMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeStaticMatmulPass)

  StringRef getArgument() const final {
    return "rair-materialize-static-matmul";
  }
  StringRef getDescription() const final {
    return "Materialize static memref matmul as a RAIR Core trace";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, rair::RAIRDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> candidates;
    module.walk([&](linalg::MatmulOp matmul) {
      if (!matmul->getParentOfType<rair::ComputeOp>())
        candidates.push_back(matmul);
    });
    if (candidates.empty()) {
      markAllAnalysesPreserved();
      return;
    }

    for (linalg::MatmulOp matmul : candidates) {
      if (failed(validateMatmul(matmul))) {
        signalPassFailure();
        return;
      }
    }

    SymbolTable symbols(module);
    if (Operation *symbol = symbols.lookup(kDefaultTargetName)) {
      if (!isa<rair::TargetOp>(symbol)) {
        module.emitError() << "symbol @" << kDefaultTargetName
                           << " already exists and is not a rair.target";
        signalPassFailure();
        return;
      }
    } else {
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToStart(module.getBody());
      builder.create<rair::TargetOp>(module.getLoc(), kDefaultTargetName,
                                     kDefaultTargetKind, IntegerAttr(),
                                     IntegerAttr());
    }

    for (linalg::MatmulOp matmul : candidates)
      materializeMatmul(matmul);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
rair::createMaterializeStaticMatmulPass() {
  return std::make_unique<MaterializeStaticMatmulPass>();
}
