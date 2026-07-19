//===- RAIRPlanAnalysis.cpp - Typed Plan DAG correctness ----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPlanAnalysis.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace rair {

TaskKind getTaskKindForStaticAction(StaticActionKind kind) {
  switch (kind) {
  case StaticActionKind::Reserve:
    return TaskKind::Reserve;
  case StaticActionKind::Move:
    return TaskKind::Move;
  case StaticActionKind::Compute:
    return TaskKind::Compute;
  case StaticActionKind::ReleaseLease:
    return TaskKind::ReleaseLease;
  }
  llvm_unreachable("unknown static action kind");
}

FailureOr<RAIRPlanGraph> RAIRPlanGraph::build(PlanOp plan) {
  RAIRPlanGraph graph(plan);
  SmallVector<TaskOp> tasks(plan.getBody().getOps<TaskOp>());
  graph.tasksBySourceAction.resize(tasks.size());
  graph.predecessors.resize(tasks.size());
  SmallVector<SmallVector<unsigned>> successors(tasks.size());

  for (TaskOp task : tasks) {
    int64_t sourceAction = task.getSourceAction();
    if (sourceAction < 0 || static_cast<size_t>(sourceAction) >= tasks.size() ||
        graph.tasksBySourceAction[sourceAction])
      return failure();
    graph.tasksBySourceAction[sourceAction] = task;
  }
  if (llvm::any_of(graph.tasksBySourceAction,
                   [](TaskOp task) { return !task; }))
    return failure();

  for (TaskOp task : tasks) {
    unsigned consumer = task.getSourceAction();
    for (Value dependency : task.getDependencies()) {
      auto producer = dependency.getDefiningOp<TaskOp>();
      if (!producer || producer->getParentOp() != plan.getOperation())
        return failure();
      int64_t producerId = producer.getSourceAction();
      if (producerId < 0 || static_cast<size_t>(producerId) >= tasks.size())
        return failure();
      graph.predecessors[consumer].push_back(producerId);
      successors[producerId].push_back(consumer);
    }
  }

  graph.reachability.assign(tasks.size(), llvm::BitVector(tasks.size()));
  for (unsigned start = 0; start < tasks.size(); ++start) {
    SmallVector<unsigned> worklist(successors[start]);
    while (!worklist.empty()) {
      unsigned next = worklist.pop_back_val();
      if (graph.reachability[start].test(next))
        continue;
      graph.reachability[start].set(next);
      llvm::append_range(worklist, successors[next]);
    }
  }
  return graph;
}

bool RAIRPlanGraph::isReachable(unsigned from, unsigned to) const {
  return from < reachability.size() && to < reachability.size() &&
         reachability[from].test(to);
}

std::optional<PlanValidationIssue>
RAIRPlanGraph::validateAgainst(const RAIRStaticEffectGraph &coreGraph) const {
  ArrayRef<StaticAction> actions = coreGraph.getActions();
  if (tasksBySourceAction.size() != actions.size())
    return PlanTaskCountMismatch{
        plan, static_cast<unsigned>(actions.size()),
        static_cast<unsigned>(tasksBySourceAction.size())};

  for (auto [actionId, action] : llvm::enumerate(actions)) {
    TaskKind expectedKind = getTaskKindForStaticAction(action.kind);
    TaskOp task = tasksBySourceAction[actionId];
    if (task.getKind() != expectedKind)
      return PlanActionKindMismatch{task, static_cast<unsigned>(actionId),
                                    expectedKind};
  }

  for (const StaticEffectEdge &edge : coreGraph.getEdges())
    if (!isReachable(edge.from, edge.to))
      return PlanMissingCorrectnessPath{plan, edge.from, edge.to};
  return std::nullopt;
}

static bool hasAlternativePath(unsigned from, unsigned to, unsigned skippedFrom,
                               unsigned skippedTo,
                               ArrayRef<SmallVector<unsigned>> successors) {
  llvm::BitVector visited(successors.size());
  SmallVector<unsigned> worklist;
  for (unsigned successor : successors[from])
    if (from != skippedFrom || successor != skippedTo)
      worklist.push_back(successor);

  while (!worklist.empty()) {
    unsigned next = worklist.pop_back_val();
    if (next == to)
      return true;
    if (visited.test(next))
      continue;
    visited.set(next);
    llvm::append_range(worklist, successors[next]);
  }
  return false;
}

StaticPlanDependencies
computeStaticTransitiveReduction(const RAIRStaticEffectGraph &graph) {
  unsigned actionCount = graph.getActions().size();
  SmallVector<SmallVector<unsigned>> successors(actionCount);
  for (const StaticEffectEdge &edge : graph.getEdges())
    successors[edge.from].push_back(edge.to);

  StaticPlanDependencies dependencies(actionCount);
  for (const StaticEffectEdge &edge : graph.getEdges())
    if (!hasAlternativePath(edge.from, edge.to, edge.from, edge.to, successors))
      dependencies[edge.to].push_back(edge.from);
  return dependencies;
}

} // namespace rair
