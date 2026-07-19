//===- PrintEffectReport.cpp - Print a RAIR static effect graph -----------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h"
#include "torch-mlir/Dialect/RAIR/Transforms/RAIRStaticEffectGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <variant>

using namespace mlir;

namespace {

static StringRef stringifyOverlap(rair::StaticOverlapKind kind) {
  switch (kind) {
  case rair::StaticOverlapKind::Disjoint:
    return "disjoint";
  case rair::StaticOverlapKind::Overlap:
    return "overlap";
  case rair::StaticOverlapKind::MayOverlap:
    return "may_overlap";
  }
  llvm_unreachable("unknown overlap kind");
}

static StringRef stringifyAccess(rair::StaticAccessKind kind) {
  switch (kind) {
  case rair::StaticAccessKind::Read:
    return "read";
  case rair::StaticAccessKind::Write:
    return "write";
  case rair::StaticAccessKind::ReadWrite:
    return "readwrite";
  }
  llvm_unreachable("unknown access kind");
}

static StringRef stringifyAction(rair::StaticActionKind kind) {
  switch (kind) {
  case rair::StaticActionKind::Reserve:
    return "reserve";
  case rair::StaticActionKind::Move:
    return "move";
  case rair::StaticActionKind::Compute:
    return "compute";
  case rair::StaticActionKind::ReleaseLease:
    return "release";
  }
  llvm_unreachable("unknown action kind");
}

static StringRef
stringifyLifetimeConstraint(rair::StaticLifetimeConstraint constraint) {
  switch (constraint) {
  case rair::StaticLifetimeConstraint::AllocateBeforeUse:
    return "allocate-before-use";
  case rair::StaticLifetimeConstraint::UseBeforeFree:
    return "use-before-free";
  case rair::StaticLifetimeConstraint::AllocateBeforeFree:
    return "allocate-before-free";
  }
  llvm_unreachable("unknown lifetime constraint");
}

static void printConflicts(llvm::raw_ostream &os,
                           rair::StaticMemoryConflict conflicts) {
  bool needsSeparator = false;
  auto printConflict = [&](rair::StaticMemoryConflict conflict,
                           StringRef spelling) {
    if (!rair::hasMemoryConflict(conflicts, conflict))
      return;
    if (needsSeparator)
      os << '+';
    os << spelling;
    needsSeparator = true;
  };
  printConflict(rair::StaticMemoryConflict::RAW, "RAW");
  printConflict(rair::StaticMemoryConflict::WAR, "WAR");
  printConflict(rair::StaticMemoryConflict::WAW, "WAW");
}

static void printArray(llvm::raw_ostream &os, ArrayRef<int64_t> values) {
  os << '[';
  llvm::interleaveComma(values, os);
  os << ']';
}

static std::string getBaseLabel(rair::ViewOp view,
                                const rair::RAIRStaticEffectGraph &graph) {
  if (auto argument = dyn_cast<BlockArgument>(view.getBase()))
    return (Twine("arg") + Twine(argument.getArgNumber())).str();
  if (auto reserve = view.getBase().getDefiningOp<rair::ReserveOp>()) {
    std::optional<unsigned> actionId =
        graph.getActionId(reserve.getOperation());
    if (actionId)
      return (Twine("a") + Twine(*actionId) + ".buffer").str();
  }
  return "external";
}

static void printFootprints(llvm::raw_ostream &os,
                            ArrayRef<rair::StaticFootprint> footprints) {
  os << '[';
  llvm::interleaveComma(footprints, os,
                        [&](const rair::StaticFootprint &footprint) {
                          os << stringifyAccess(footprint.access) << "(r"
                             << footprint.region << ')';
                        });
  os << ']';
}

static void printReason(llvm::raw_ostream &os,
                        const rair::StaticEdgeReason &reason) {
  if (auto memory = std::get_if<rair::StaticMemoryEdgeReason>(&reason)) {
    os << "memory conflict=";
    printConflicts(os, memory->conflicts);
    os << ' ' << stringifyAccess(memory->earlierAccess) << "(r"
       << memory->earlierRegion << ") -> "
       << stringifyAccess(memory->laterAccess) << "(r" << memory->laterRegion
       << ") relation=" << stringifyOverlap(memory->overlap);
    return;
  }

  const auto &lifetime = std::get<rair::StaticLifetimeEdgeReason>(reason);
  os << "lifetime " << stringifyLifetimeConstraint(lifetime.constraint)
     << " lease=a" << lifetime.reserveAction;
}

static void printScopeReport(func::FuncOp function, unsigned scopeIndex,
                             const rair::RAIRStaticEffectGraph &graph,
                             llvm::raw_ostream &os) {
  rair::ScopeOp scope = graph.getScope();
  os << "RAIR effect report: func @" << function.getSymName() << " scope "
     << scopeIndex << " target @" << scope.getTargetAttr().getValue() << '\n';
  os << "regions:\n";
  for (auto [regionId, storedView] : llvm::enumerate(graph.getRegions())) {
    rair::ViewOp view = storedView;
    auto type = cast<MemRefType>(view.getBase().getType());
    auto space = cast<rair::MemorySpaceAttr>(type.getMemorySpace());
    os << "  r" << regionId << " base=" << getBaseLabel(view, graph)
       << " space=" << rair::stringifyMemorySpace(space.getValue())
       << " offsets=";
    printArray(os, view.getOffsets());
    os << " sizes=";
    printArray(os, view.getSizes());
    os << " strides=";
    printArray(os, view.getStrides());
    os << '\n';
  }

  os << "relations:\n";
  for (const rair::StaticRegionRelation &relation : graph.getRelations())
    os << "  r" << relation.lhs << " vs r" << relation.rhs << " = "
       << stringifyOverlap(relation.overlap) << '\n';

  os << "actions:\n";
  for (auto [actionId, action] : llvm::enumerate(graph.getActions())) {
    os << "  a" << actionId << ' ' << stringifyAction(action.kind);
    if (auto reserve = dyn_cast<rair::ReserveOp>(action.operation))
      os << " space=" << rair::stringifyMemorySpace(reserve.getSpace());
    if (!action.footprints.empty()) {
      os << " effects=";
      printFootprints(os, action.footprints);
    }
    if (auto release = dyn_cast<rair::ReleaseLeaseOp>(action.operation)) {
      auto reserve = release.getLease().getDefiningOp<rair::ReserveOp>();
      std::optional<unsigned> reserveId =
          graph.getActionId(reserve.getOperation());
      assert(reserveId && "verified release must reference a graph reserve");
      os << " lease=a" << *reserveId;
    }
    os << '\n';
  }

  os << "edges:\n";
  for (const rair::StaticEffectEdge &edge : graph.getEdges()) {
    os << "  a" << edge.from << " -> a" << edge.to << ' ';
    llvm::interleave(
        edge.reasons,
        [&](const rair::StaticEdgeReason &reason) { printReason(os, reason); },
        [&]() { os << "; "; });
    os << '\n';
  }

  os << "independent:\n";
  for (unsigned lhs = 0; lhs < graph.getActions().size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < graph.getActions().size(); ++rhs)
      if (!graph.hasEdge(lhs, rhs))
        os << "  a" << lhs << " || a" << rhs << '\n';

  os << "summary: regions=" << graph.getRegions().size()
     << " actions=" << graph.getActions().size()
     << " graph_edges=" << graph.getEdges().size() << "\n\n";
}

class InferEffectsPass
    : public PassWrapper<InferEffectsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferEffectsPass)

  StringRef getArgument() const final { return "rair-infer-effects"; }
  StringRef getDescription() const final {
    return "Infer and report static RAIR Core effects";
  }

  void runOnOperation() final {
    bool failedToBuildGraph = false;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      unsigned scopeIndex = 0;
      function.walk([&](rair::ScopeOp scope) {
        FailureOr<rair::RAIRStaticEffectGraph> graph =
            rair::RAIRStaticEffectGraph::build(scope);
        if (failed(graph)) {
          scope.emitError("could not build a static effect graph from "
                          "verified Core IR");
          failedToBuildGraph = true;
          return;
        }
        printScopeReport(function, scopeIndex++, *graph, llvm::outs());
      });
    }
    if (failedToBuildGraph) {
      signalPassFailure();
      return;
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> rair::createInferEffectsPass() {
  return std::make_unique<InferEffectsPass>();
}
