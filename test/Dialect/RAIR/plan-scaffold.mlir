// RUN: torch-mlir-opt %s | FileCheck %s
// RUN: torch-mlir-opt %s --canonicalize --cse | FileCheck %s
// RUN: torch-mlir-opt %s --rair-materialize-plan | FileCheck %s

// CHECK: rair.target @gemmini
// CHECK-LABEL: func.func @hand_written_plan
// CHECK: rair.scope @gemmini {
// CHECK: %[[BUFFER:.*]], %[[LEASE:.*]] = rair.reserve
// CHECK: rair.release_lease %[[LEASE]], %[[BUFFER]]
// CHECK: rair.plan @gemmini {
// CHECK: %[[RESERVE:.*]] = rair.task() {kind = #rair.task_kind<reserve>, source_action = 0 : i64} : () -> !rair.event<@gemmini>
// CHECK: rair.task(%[[RESERVE]]) {kind = #rair.task_kind<release_lease>, source_action = 1 : i64} : (!rair.event<@gemmini>) -> !rair.event<@gemmini>

rair.target @gemmini {kind = "gemmini"}

func.func @hand_written_plan() {
  rair.scope @gemmini {
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4xf32, #rair.space<spad>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4xf32, #rair.space<spad>>
    rair.plan @gemmini {
      %reserve = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      %release = rair.task (%reserve) {
        kind = #rair.task_kind<release_lease>, source_action = 1 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
    }
  }
  return
}
