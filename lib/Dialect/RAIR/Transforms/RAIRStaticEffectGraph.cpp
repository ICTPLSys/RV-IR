//===- RAIRStaticEffectGraph.cpp - Typed static Core effect graph --------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRStaticEffectGraph.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <tuple>

using namespace mlir;

namespace rair {

StaticMemoryConflict operator|(StaticMemoryConflict lhs,
                               StaticMemoryConflict rhs) {
  return static_cast<StaticMemoryConflict>(static_cast<uint8_t>(lhs) |
                                           static_cast<uint8_t>(rhs));
}

bool hasMemoryConflict(StaticMemoryConflict conflicts,
                       StaticMemoryConflict conflict) {
  return (static_cast<uint8_t>(conflicts) & static_cast<uint8_t>(conflict)) !=
         0;
}

static bool writesMemory(StaticAccessKind kind) {
  return kind == StaticAccessKind::Write || kind == StaticAccessKind::ReadWrite;
}

static bool readsMemory(StaticAccessKind kind) {
  return kind == StaticAccessKind::Read || kind == StaticAccessKind::ReadWrite;
}

static StaticMemoryConflict classifyMemoryConflict(StaticAccessKind earlier,
                                                   StaticAccessKind later) {
  StaticMemoryConflict conflicts = StaticMemoryConflict::None;
  if (writesMemory(earlier) && readsMemory(later))
    conflicts = conflicts | StaticMemoryConflict::RAW;
  if (readsMemory(earlier) && writesMemory(later))
    conflicts = conflicts | StaticMemoryConflict::WAR;
  if (writesMemory(earlier) && writesMemory(later))
    conflicts = conflicts | StaticMemoryConflict::WAW;
  return conflicts;
}

StaticOverlapKind classifyStaticOverlap(ViewOp lhs, ViewOp rhs) {
  if (lhs == rhs)
    return StaticOverlapKind::Overlap;

  if (lhs.getBase() != rhs.getBase()) {
    // Every rair.reserve creates a fresh allocation. Function arguments and
    // other externally supplied bases may alias unless their physical spaces
    // make aliasing impossible.
    if (lhs.getBase().getDefiningOp<ReserveOp>() ||
        rhs.getBase().getDefiningOp<ReserveOp>())
      return StaticOverlapKind::Disjoint;
    auto lhsType = cast<MemRefType>(lhs.getBase().getType());
    auto rhsType = cast<MemRefType>(rhs.getBase().getType());
    auto lhsSpace = cast<MemorySpaceAttr>(lhsType.getMemorySpace()).getValue();
    auto rhsSpace = cast<MemorySpaceAttr>(rhsType.getMemorySpace()).getValue();
    if (lhsSpace != rhsSpace && stringifyMemorySpace(lhsSpace) != "unknown" &&
        stringifyMemorySpace(rhsSpace) != "unknown")
      return StaticOverlapKind::Disjoint;
    return StaticOverlapKind::MayOverlap;
  }

  if (lhs.getOffsets() == rhs.getOffsets() &&
      lhs.getSizes() == rhs.getSizes() && lhs.getStrides() == rhs.getStrides())
    return StaticOverlapKind::Overlap;

  bool allUnitStride = true;
  for (unsigned dim = 0; dim < lhs.getSizes().size(); ++dim) {
    int64_t lhsFirst = lhs.getOffsets()[dim];
    int64_t rhsFirst = rhs.getOffsets()[dim];
    int64_t lhsLast =
        lhsFirst + (lhs.getSizes()[dim] - 1) * lhs.getStrides()[dim];
    int64_t rhsLast =
        rhsFirst + (rhs.getSizes()[dim] - 1) * rhs.getStrides()[dim];
    if (lhsLast < rhsFirst || rhsLast < lhsFirst)
      return StaticOverlapKind::Disjoint;
    allUnitStride &= lhs.getStrides()[dim] == 1 && rhs.getStrides()[dim] == 1;
  }

  // Bounding boxes that overlap are exact for unit-stride rectangles. For
  // strided regions v0.1 deliberately stays conservative.
  return allUnitStride ? StaticOverlapKind::Overlap
                       : StaticOverlapKind::MayOverlap;
}

static void mergeFootprint(StaticAction &action, unsigned region,
                           StaticAccessKind access) {
  auto iterator =
      llvm::find_if(action.footprints, [&](const StaticFootprint &footprint) {
        return footprint.region == region;
      });
  if (iterator == action.footprints.end()) {
    action.footprints.push_back({region, access});
    return;
  }
  if (iterator->access != access)
    iterator->access = StaticAccessKind::ReadWrite;
}

static bool reasonsEqual(const StaticEdgeReason &lhs,
                         const StaticEdgeReason &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (auto lhsMemory = std::get_if<StaticMemoryEdgeReason>(&lhs)) {
    auto rhsMemory = std::get_if<StaticMemoryEdgeReason>(&rhs);
    return lhsMemory->conflicts == rhsMemory->conflicts &&
           lhsMemory->earlierAccess == rhsMemory->earlierAccess &&
           lhsMemory->laterAccess == rhsMemory->laterAccess &&
           lhsMemory->earlierRegion == rhsMemory->earlierRegion &&
           lhsMemory->laterRegion == rhsMemory->laterRegion &&
           lhsMemory->overlap == rhsMemory->overlap;
  }
  auto lhsLifetime = std::get<StaticLifetimeEdgeReason>(lhs);
  auto rhsLifetime = std::get<StaticLifetimeEdgeReason>(rhs);
  return lhsLifetime.constraint == rhsLifetime.constraint &&
         lhsLifetime.reserveAction == rhsLifetime.reserveAction;
}

static unsigned getConflictSortRank(StaticMemoryConflict conflicts) {
  switch (static_cast<uint8_t>(conflicts)) {
  case static_cast<uint8_t>(StaticMemoryConflict::RAW):
    return 0;
  case static_cast<uint8_t>(StaticMemoryConflict::RAW) |
      static_cast<uint8_t>(StaticMemoryConflict::WAR):
    return 1;
  case static_cast<uint8_t>(StaticMemoryConflict::RAW) |
      static_cast<uint8_t>(StaticMemoryConflict::WAR) |
      static_cast<uint8_t>(StaticMemoryConflict::WAW):
    return 2;
  case static_cast<uint8_t>(StaticMemoryConflict::RAW) |
      static_cast<uint8_t>(StaticMemoryConflict::WAW):
    return 3;
  case static_cast<uint8_t>(StaticMemoryConflict::WAR):
    return 4;
  case static_cast<uint8_t>(StaticMemoryConflict::WAR) |
      static_cast<uint8_t>(StaticMemoryConflict::WAW):
    return 5;
  case static_cast<uint8_t>(StaticMemoryConflict::WAW):
    return 6;
  default:
    llvm_unreachable("memory edge must contain a conflict");
  }
}

static unsigned getAccessSortRank(StaticAccessKind access) {
  switch (access) {
  case StaticAccessKind::Read:
    return 0;
  case StaticAccessKind::ReadWrite:
    return 1;
  case StaticAccessKind::Write:
    return 2;
  }
  llvm_unreachable("unknown access kind");
}

static unsigned getLifetimeSortRank(StaticLifetimeConstraint constraint) {
  switch (constraint) {
  case StaticLifetimeConstraint::AllocateBeforeFree:
    return 0;
  case StaticLifetimeConstraint::AllocateBeforeUse:
    return 1;
  case StaticLifetimeConstraint::UseBeforeFree:
    return 2;
  }
  llvm_unreachable("unknown lifetime constraint");
}

static auto getReasonSortKey(const StaticEdgeReason &reason) {
  if (auto memory = std::get_if<StaticMemoryEdgeReason>(&reason))
    return std::make_tuple(0u, getConflictSortRank(memory->conflicts),
                           getAccessSortRank(memory->earlierAccess),
                           memory->earlierRegion, memory->laterRegion);
  auto lifetime = std::get<StaticLifetimeEdgeReason>(reason);
  return std::make_tuple(1u, getLifetimeSortRank(lifetime.constraint),
                         lifetime.reserveAction, 0u, 0u);
}

FailureOr<RAIRStaticEffectGraph> RAIRStaticEffectGraph::build(ScopeOp scope) {
  RAIRStaticEffectGraph graph(scope);
  llvm::DenseMap<Operation *, unsigned> regionIds;
  llvm::DenseMap<Operation *, unsigned> actionIds;

  for (Operation &operation : scope.getBody().front()) {
    if (auto view = dyn_cast<ViewOp>(operation)) {
      regionIds[view.getOperation()] = graph.regions.size();
      graph.regions.push_back(view);
      continue;
    }

    StaticActionKind actionKind;
    if (isa<ReserveOp>(operation))
      actionKind = StaticActionKind::Reserve;
    else if (isa<MoveOp>(operation))
      actionKind = StaticActionKind::Move;
    else if (isa<ComputeOp>(operation))
      actionKind = StaticActionKind::Compute;
    else if (isa<ReleaseLeaseOp>(operation))
      actionKind = StaticActionKind::ReleaseLease;
    else
      continue;

    StaticAction action{&operation, actionKind, {}};
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(&operation);
    if (!effectInterface)
      return failure();
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectInterface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value value = effect.getValue();
      if (!value)
        continue;
      auto view = value.getDefiningOp<ViewOp>();
      if (!view)
        continue;
      auto regionIterator = regionIds.find(view.getOperation());
      if (regionIterator == regionIds.end())
        return failure();
      if (isa<MemoryEffects::Read>(effect.getEffect()))
        mergeFootprint(action, regionIterator->second, StaticAccessKind::Read);
      else if (isa<MemoryEffects::Write>(effect.getEffect()))
        mergeFootprint(action, regionIterator->second, StaticAccessKind::Write);
    }

    actionIds[&operation] = graph.actions.size();
    graph.actions.push_back(std::move(action));
  }

  for (unsigned lhs = 0; lhs < graph.regions.size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < graph.regions.size(); ++rhs)
      graph.relations.push_back(
          {lhs, rhs,
           classifyStaticOverlap(graph.regions[lhs], graph.regions[rhs])});

  auto addEdge = [&](unsigned from, unsigned to, StaticEdgeReason reason) {
    auto edge = llvm::find_if(graph.edges, [&](const StaticEffectEdge &edge) {
      return edge.from == from && edge.to == to;
    });
    if (edge == graph.edges.end()) {
      graph.edges.push_back({from, to, {std::move(reason)}});
      return;
    }
    if (!llvm::any_of(edge->reasons, [&](const StaticEdgeReason &existing) {
          return reasonsEqual(existing, reason);
        }))
      edge->reasons.push_back(std::move(reason));
  };

  for (unsigned earlierId = 0; earlierId < graph.actions.size(); ++earlierId) {
    for (unsigned laterId = earlierId + 1; laterId < graph.actions.size();
         ++laterId) {
      for (const StaticFootprint &earlier :
           graph.actions[earlierId].footprints) {
        for (const StaticFootprint &later : graph.actions[laterId].footprints) {
          if (!writesMemory(earlier.access) && !writesMemory(later.access))
            continue;
          StaticOverlapKind overlap =
              graph.getOverlap(earlier.region, later.region);
          if (overlap == StaticOverlapKind::Disjoint)
            continue;
          addEdge(earlierId, laterId,
                  StaticMemoryEdgeReason{
                      classifyMemoryConflict(earlier.access, later.access),
                      earlier.access, later.access, earlier.region,
                      later.region, overlap});
        }
      }
    }
  }

  for (unsigned reserveId = 0; reserveId < graph.actions.size(); ++reserveId) {
    auto reserve = dyn_cast<ReserveOp>(graph.actions[reserveId].operation);
    if (!reserve)
      continue;
    if (!reserve.getLease().hasOneUse())
      return failure();
    auto release =
        dyn_cast<ReleaseLeaseOp>(*reserve.getLease().getUsers().begin());
    if (!release)
      return failure();
    auto releaseIterator = actionIds.find(release.getOperation());
    if (releaseIterator == actionIds.end())
      return failure();
    unsigned releaseId = releaseIterator->second;

    addEdge(reserveId, releaseId,
            StaticLifetimeEdgeReason{
                StaticLifetimeConstraint::AllocateBeforeFree, reserveId});
    for (unsigned actionId = reserveId + 1; actionId < releaseId; ++actionId) {
      bool touchesBuffer =
          llvm::any_of(graph.actions[actionId].footprints,
                       [&](const StaticFootprint &footprint) {
                         ViewOp view = graph.regions[footprint.region];
                         return view.getBase() == reserve.getBuffer();
                       });
      if (!touchesBuffer)
        continue;
      addEdge(reserveId, actionId,
              StaticLifetimeEdgeReason{
                  StaticLifetimeConstraint::AllocateBeforeUse, reserveId});
      addEdge(actionId, releaseId,
              StaticLifetimeEdgeReason{StaticLifetimeConstraint::UseBeforeFree,
                                       reserveId});
    }
  }

  llvm::sort(graph.edges,
             [](const StaticEffectEdge &lhs, const StaticEffectEdge &rhs) {
               return std::tie(lhs.from, lhs.to) < std::tie(rhs.from, rhs.to);
             });
  for (StaticEffectEdge &edge : graph.edges)
    llvm::sort(edge.reasons,
               [](const StaticEdgeReason &lhs, const StaticEdgeReason &rhs) {
                 return getReasonSortKey(lhs) < getReasonSortKey(rhs);
               });

  return graph;
}

std::optional<unsigned> RAIRStaticEffectGraph::getRegionId(ViewOp view) const {
  for (auto [index, candidate] : llvm::enumerate(regions))
    if (candidate == view)
      return index;
  return std::nullopt;
}

std::optional<unsigned>
RAIRStaticEffectGraph::getActionId(Operation *operation) const {
  for (auto [index, action] : llvm::enumerate(actions))
    if (action.operation == operation)
      return index;
  return std::nullopt;
}

StaticOverlapKind RAIRStaticEffectGraph::getOverlap(unsigned lhs,
                                                    unsigned rhs) const {
  if (lhs == rhs)
    return StaticOverlapKind::Overlap;
  if (lhs > rhs)
    std::swap(lhs, rhs);
  auto relation =
      llvm::find_if(relations, [&](const StaticRegionRelation &relation) {
        return relation.lhs == lhs && relation.rhs == rhs;
      });
  assert(relation != relations.end() && "invalid static region IDs");
  return relation->overlap;
}

bool RAIRStaticEffectGraph::hasEdge(unsigned from, unsigned to) const {
  return llvm::any_of(edges, [&](const StaticEffectEdge &edge) {
    return edge.from == from && edge.to == to;
  });
}

} // namespace rair
