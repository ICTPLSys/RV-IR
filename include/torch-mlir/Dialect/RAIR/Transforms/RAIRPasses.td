//===-- RAIRPasses.td - RAIR transformation passes -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES
#define TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES

include "mlir/Pass/PassBase.td"

def VerifyRAIRLifetimesPass
    : Pass<"rair-verify-lifetimes", "func::FuncOp"> {
  let summary = "Verify straight-line RAIR context and buffer lifetimes";
  let constructor = "rair::createVerifyLifetimesPass()";
  let dependentDialects = ["rair::RAIRDialect", "func::FuncDialect"];
  let description = [{
    Verifies the Phase 0 linear ownership contract for RAIR resources. Each
    context produced by rair.acquire must be released exactly once, each buffer
    produced by rair.alloc_buffer must be deallocated exactly once using the
    same context, and neither resource may be used after release.

    Resource lifetimes that cross calls, returns, branches, loops, nested
    regions, or derived memref aliases are rejected until a control-flow-aware
    ownership analysis is introduced.
  }];
}

def InferRAIREffectsPass
    : Pass<"rair-infer-effects", "ModuleOp"> {
  let summary = "Infer and report static RAIR Core effects";
  let constructor = "rair::createInferEffectsPass()";
  let dependentDialects = ["rair::RAIRDialect", "func::FuncDialect"];
  let description = [{
    An analysis-only pass for RAIR Core v0.1. It classifies pairs of static
    rair.view regions as disjoint, overlap, or may-overlap; extracts the memory
    and lease-lifetime footprint of reserve, move, compute, and release actions
    from their declared effect interfaces; and prints the conflict edges
    required by the Core reference trace.

    The pass does not mutate IR, materialize RAIR Plan tasks, select a schedule,
    or invoke a target backend.
  }];
}

def MaterializeStaticMatmulPass
    : Pass<"rair-materialize-static-matmul", "ModuleOp"> {
  let summary = "Materialize static memref matmul as a RAIR Core trace";
  let constructor = "rair::createMaterializeStaticMatmulPass()";
  let dependentDialects = ["rair::RAIRDialect", "linalg::LinalgDialect",
                           "func::FuncDialect"];
  let description = [{
    Replaces each eligible static memref linalg.matmul with one self-contained
    RAIR Core v0.1 reference trace: external views, SPAD/ACC reservations,
    input and initial-output moves, a compute wrapper retaining the original
    Linalg payload, output writeback, and reverse-order lease release.

    The pass creates or reuses the module target @rair_default. It is an
    independent opt-in Core producer and does not invoke or change the legacy
    --convert-linalg-to-rair pipeline, materialize Plan IR, or add a backend.
  }];
}

def MaterializeRAIRPlanPass
    : Pass<"rair-materialize-plan", "ModuleOp"> {
  let summary = "Materialize a raw correctness Plan for each RAIR Core scope";
  let constructor = "rair::createMaterializePlanPass()";
  let dependentDialects = ["rair::RAIRDialect", "func::FuncDialect"];
  let description = [{
    Builds the reusable RAIRStaticEffectGraph for each Core scope and appends
    one structurally associated rair.plan. Every Core action becomes one typed
    task with the same textual-order source action ID, and every raw graph edge
    becomes a target-aware event dependency.

    Existing plans are accepted only when their action kinds and dependency
    sets exactly match the current raw graph, making the pass idempotent while
    diagnosing stale plans. The pass does not perform transitive reduction,
    capacity or queue scheduling, runtime lowering, or backend selection.
  }];
}

def VerifyRAIRPlanPass
    : Pass<"rair-verify-plan", "ModuleOp"> {
  let summary = "Verify RAIR Plan correctness reachability";
  let constructor = "rair::createVerifyPlanPass()";
  let dependentDialects = ["rair::RAIRDialect", "func::FuncDialect"];
  let description = [{
    Builds typed Core and Plan graphs, verifies one task of the matching kind
    for every source action, and proves that every raw Core correctness edge is
    preserved by a same-direction reachable path in the Plan DAG. Extra or
    transitively reduced Plan edges are legal. The pass is analysis-only.
  }];
}

#endif // TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES
