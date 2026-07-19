// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics

rair.target @gemmini {kind = "gemmini"}
rair.target @other {kind = "other"}

func.func @plan_target_mismatch() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.plan' op targets @other but containing scope targets @gemmini}}
    rair.plan @other {
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}
rair.target @other {kind = "other"}

func.func @wrong_result_target() {
  rair.scope @gemmini {
    rair.plan @gemmini {
      // expected-error @+1 {{'rair.task' op produces event for @other but containing plan targets @gemmini}}
      %event = rair.task () {
        kind = #rair.task_kind<move>, source_action = 0 : i64
      } : () -> !rair.event<@other>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}
rair.target @other {kind = "other"}

func.func @wrong_dependency_target(%foreign: !rair.event<@other>) {
  rair.scope @gemmini {
    rair.plan @gemmini {
      // expected-error @+1 {{'rair.task' op has dependency event for @other but containing plan targets @gemmini}}
      %event = rair.task (%foreign) {
        kind = #rair.task_kind<move>, source_action = 0 : i64
      } : (!rair.event<@other>) -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @external_dependency(%foreign: !rair.event<@gemmini>) {
  rair.scope @gemmini {
    rair.plan @gemmini {
      // expected-error @+1 {{'rair.task' op requires every dependency event to be produced by a task in the same rair.plan}}
      %event = rair.task (%foreign) {
        kind = #rair.task_kind<move>, source_action = 0 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @duplicate_dependency() {
  rair.scope @gemmini {
    rair.plan @gemmini {
      %first = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      // expected-error @+1 {{'rair.task' op lists the same dependency event more than once}}
      %second = rair.task (%first, %first) {
        kind = #rair.task_kind<move>, source_action = 1 : i64
      } : (!rair.event<@gemmini>, !rair.event<@gemmini>)
        -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @duplicate_source_action() {
  rair.scope @gemmini {
    rair.plan @gemmini {
      %first = rair.task () {
        kind = #rair.task_kind<reserve>, source_action = 0 : i64
      } : () -> !rair.event<@gemmini>
      // expected-error @+1 {{'rair.task' op duplicates source_action ID 0 in the containing rair.plan}}
      %second = rair.task (%first) {
        kind = #rair.task_kind<move>, source_action = 0 : i64
      } : (!rair.event<@gemmini>) -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @negative_source_action() {
  rair.scope @gemmini {
    rair.plan @gemmini {
      // expected-error @+1 {{'rair.task' op requires a non-negative source_action ID}}
      %event = rair.task () {
        kind = #rair.task_kind<compute>, source_action = -1 : i64
      } : () -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @non_dense_source_actions() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.plan' op requires dense source_action IDs [0, 1), but ID 0 is missing}}
    rair.plan @gemmini {
      %event = rair.task () {
        kind = #rair.task_kind<compute>, source_action = 1 : i64
      } : () -> !rair.event<@gemmini>
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @non_plan_operation() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.plan' op contains non-Plan operation arith.constant directly in its task graph}}
    rair.plan @gemmini {
      %zero = arith.constant 0 : i32
    }
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @plan_not_last(%arg: memref<4xf32, #rair.space<host>>) {
  // expected-error @+1 {{'rair.scope' op requires its associated rair.plan after every Core operation}}
  rair.scope @gemmini {
    rair.plan @gemmini {
    }
    %view = rair.view %arg {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<4xf32, #rair.space<host>> -> !rair.region
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @multiple_plans() {
  // expected-error @+1 {{'rair.scope' op allows at most one associated rair.plan}}
  rair.scope @gemmini {
    rair.plan @gemmini {
    }
    rair.plan @gemmini {
    }
  }
  return
}
