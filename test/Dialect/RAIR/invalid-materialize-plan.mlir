// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics --rair-materialize-plan

rair.target @gemmini {kind = "gemmini"}

func.func @task_count_mismatch() {
  rair.scope @gemmini {
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4xf32, #rair.space<spad>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4xf32, #rair.space<spad>>
    // expected-error @+1 {{'rair.plan' op has 1 tasks, but its source Core graph has 2 actions}}
    rair.plan @gemmini {
      %reserve = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @task_kind_mismatch() {
  rair.scope @gemmini {
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4xf32, #rair.space<spad>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4xf32, #rair.space<spad>>
    rair.plan @gemmini {
      // expected-error @+1 {{'rair.task' op kind #rair.task_kind<move> does not match source_action 0 kind reserve}}
      %reserve = rair.task () {
        kind = #rair.task_kind<move>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      %release = rair.task (%reserve) {
        kind = #rair.task_kind<release_lease>, source_action = 1 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @task_dependency_mismatch() {
  rair.scope @gemmini {
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4xf32, #rair.space<spad>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4xf32, #rair.space<spad>>
    // expected-error @+1 {{'rair.plan' op does not preserve required Core correctness path from source_action 0 to source_action 1}}
    rair.plan @gemmini {
      %reserve = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      %release = rair.task () {
        kind = #rair.task_kind<release_lease>, source_action = 1 : i64
      } : () -> !rair.event<@gemmini>
    }
  }
  return
}
