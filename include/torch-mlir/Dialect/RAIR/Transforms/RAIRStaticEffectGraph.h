//===- RAIRStaticEffectGraph.h - Typed static Core effect graph -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRSTATICEFFECTGRAPH_H
#define TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRSTATICEFFECTGRAPH_H

#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace rair {

enum class StaticOverlapKind { Disjoint, Overlap, MayOverlap };
enum class StaticAccessKind { Read, Write, ReadWrite };
enum class StaticActionKind { Reserve, Move, Compute, ReleaseLease };

enum class StaticMemoryConflict : uint8_t {
  None = 0,
  RAW = 1 << 0,
  WAR = 1 << 1,
  WAW = 1 << 2,
};

StaticMemoryConflict operator|(StaticMemoryConflict lhs,
                               StaticMemoryConflict rhs);
bool hasMemoryConflict(StaticMemoryConflict conflicts,
                       StaticMemoryConflict conflict);

enum class StaticLifetimeConstraint {
  AllocateBeforeUse,
  UseBeforeFree,
  AllocateBeforeFree,
};

struct StaticRegionRelation {
  unsigned lhs;
  unsigned rhs;
  StaticOverlapKind overlap;
};

struct StaticFootprint {
  unsigned region;
  StaticAccessKind access;
};

struct StaticAction {
  mlir::Operation *operation;
  StaticActionKind kind;
  llvm::SmallVector<StaticFootprint> footprints;
};

struct StaticMemoryEdgeReason {
  StaticMemoryConflict conflicts;
  StaticAccessKind earlierAccess;
  StaticAccessKind laterAccess;
  unsigned earlierRegion;
  unsigned laterRegion;
  StaticOverlapKind overlap;
};

struct StaticLifetimeEdgeReason {
  StaticLifetimeConstraint constraint;
  unsigned reserveAction;
};

using StaticEdgeReason =
    std::variant<StaticMemoryEdgeReason, StaticLifetimeEdgeReason>;

struct StaticEffectEdge {
  unsigned from;
  unsigned to;
  llvm::SmallVector<StaticEdgeReason> reasons;
};

/// Conservatively classifies two verified static rair.view descriptors.
StaticOverlapKind classifyStaticOverlap(ViewOp lhs, ViewOp rhs);

/// Immutable, scope-local correctness graph for a verified Core v0.1 trace.
/// IDs are stable textual-order indices into the returned arrays. Stored op
/// handles remain valid only while the analyzed scope is not mutated.
class RAIRStaticEffectGraph {
public:
  static mlir::FailureOr<RAIRStaticEffectGraph> build(ScopeOp scope);

  ScopeOp getScope() const { return scope; }
  llvm::ArrayRef<ViewOp> getRegions() const { return regions; }
  llvm::ArrayRef<StaticRegionRelation> getRelations() const {
    return relations;
  }
  llvm::ArrayRef<StaticAction> getActions() const { return actions; }
  llvm::ArrayRef<StaticEffectEdge> getEdges() const { return edges; }

  std::optional<unsigned> getRegionId(ViewOp view) const;
  std::optional<unsigned> getActionId(mlir::Operation *operation) const;
  StaticOverlapKind getOverlap(unsigned lhs, unsigned rhs) const;
  bool hasEdge(unsigned from, unsigned to) const;

private:
  explicit RAIRStaticEffectGraph(ScopeOp scope) : scope(scope) {}

  ScopeOp scope;
  llvm::SmallVector<ViewOp> regions;
  llvm::SmallVector<StaticRegionRelation> relations;
  llvm::SmallVector<StaticAction> actions;
  llvm::SmallVector<StaticEffectEdge> edges;
};

} // namespace rair

#endif // TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRSTATICEFFECTGRAPH_H
