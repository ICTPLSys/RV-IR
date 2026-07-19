// RUN: torch-mlir-opt %s --rair-verify-plan -o /dev/null
// RUN: torch-mlir-opt %s --rair-materialize-plan --rair-verify-plan | FileCheck %s

// CHECK-LABEL: func.func @reduced_plan_preserves_raw_edges
// CHECK: rair.plan @gemmini {
// CHECK: %[[A0:.*]] = rair.task() {kind = #rair.task_kind<reserve>, source_action = 0 : i64}
// CHECK: %[[A1:.*]] = rair.task(%[[A0]]) {kind = #rair.task_kind<move>, source_action = 1 : i64}
// CHECK: rair.task(%[[A1]]) {kind = #rair.task_kind<release_lease>, source_action = 2 : i64}

rair.target @gemmini {kind = "gemmini"}

func.func @reduced_plan_preserves_raw_edges(
    %src: memref<4xf32, #rair.space<host>>) {
  rair.scope @gemmini {
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4xf32, #rair.space<spad>>, !rair.lease
    %src_view = rair.view %src {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<4xf32, #rair.space<host>> -> !rair.region
    %local_view = rair.view %buffer {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<4xf32, #rair.space<spad>> -> !rair.region
    rair.move %src_view to %local_view : !rair.region, !rair.region
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4xf32, #rair.space<spad>>
    rair.plan @gemmini {
      %reserve = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      %move = rair.task (%reserve) {
        kind = #rair.task_kind<move>, source_action = 1 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
      %release = rair.task (%move) {
        kind = #rair.task_kind<release_lease>, source_action = 2 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
    }
  }
  return
}
