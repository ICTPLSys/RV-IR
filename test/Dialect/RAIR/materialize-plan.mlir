// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-materialize-plan | FileCheck %s --check-prefix=PLAN
// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-materialize-plan --rair-materialize-plan | FileCheck %s --check-prefix=IDEMPOTENT
// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-materialize-plan --rair-verify-plan -o /dev/null
// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-materialize-plan --rair-infer-effects -o /dev/null | FileCheck %s --check-prefix=REPORT

// PLAN: rair.target @rair_default
// PLAN-LABEL: func.func @materialize_plan
// PLAN: rair.scope @rair_default {
// PLAN: rair.compute
// PLAN: rair.release_lease
// PLAN: rair.plan @rair_default {
// PLAN: %[[A0:.*]] = rair.task() {kind = #rair.task_kind<reserve>, source_action = 0 : i64} : () -> !rair.event<@rair_default>
// PLAN: %[[A1:.*]] = rair.task() {kind = #rair.task_kind<reserve>, source_action = 1 : i64} : () -> !rair.event<@rair_default>
// PLAN: %[[A2:.*]] = rair.task() {kind = #rair.task_kind<reserve>, source_action = 2 : i64} : () -> !rair.event<@rair_default>
// PLAN: %[[A3:.*]] = rair.task(%[[A0]]) {kind = #rair.task_kind<move>, source_action = 3 : i64}
// PLAN: %[[A4:.*]] = rair.task(%[[A1]]) {kind = #rair.task_kind<move>, source_action = 4 : i64}
// PLAN: %[[A5:.*]] = rair.task(%[[A2]]) {kind = #rair.task_kind<move>, source_action = 5 : i64}
// PLAN: %[[A6:.*]] = rair.task(%[[A3]], %[[A4]], %[[A5]]) {kind = #rair.task_kind<compute>, source_action = 6 : i64}
// PLAN: %[[A7:.*]] = rair.task(%[[A6]]) {kind = #rair.task_kind<move>, source_action = 7 : i64}
// PLAN: %[[A8:.*]] = rair.task(%[[A7]]) {kind = #rair.task_kind<release_lease>, source_action = 8 : i64}
// PLAN: %[[A9:.*]] = rair.task(%[[A6]]) {kind = #rair.task_kind<release_lease>, source_action = 9 : i64}
// PLAN: %[[A10:.*]] = rair.task(%[[A6]]) {kind = #rair.task_kind<release_lease>, source_action = 10 : i64}

// IDEMPOTENT-COUNT-1: rair.plan @rair_default
// IDEMPOTENT-COUNT-11: rair.task

// REPORT-LABEL: RAIR effect report: func @materialize_plan scope 0 target @rair_default
// REPORT: summary: regions=6 actions=11 graph_edges=24

func.func @materialize_plan(
    %a: memref<16x32xf32, #rair.space<host>>,
    %b: memref<32x16xf32, #rair.space<host>>,
    %c: memref<16x16xf32, #rair.space<host>>) {
  linalg.matmul
    ins(%a, %b
      : memref<16x32xf32, #rair.space<host>>,
        memref<32x16xf32, #rair.space<host>>)
    outs(%c : memref<16x16xf32, #rair.space<host>>)
  return
}
