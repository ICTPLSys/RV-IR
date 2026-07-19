// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics

rair.target @gemmini {kind = "gemmini"}

func.func @view_out_of_bounds(
    %arg: memref<4x4xf32, #rair.space<host>>) {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.view' op region is out of bounds at dimension 0}}
    %bad = rair.view %arg {
      offsets = array<i64: 3, 0>, sizes = array<i64: 2, 4>,
      strides = array<i64: 1, 1>
    } : memref<4x4xf32, #rair.space<host>> -> !rair.region
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @reserve_legacy_space() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.reserve' op requires #rair.space<device|spad|acc> for a reserved buffer}}
    %buffer, %lease = rair.reserve {space = #rair.space<lmem>}
      : memref<4x4xf32, #rair.space<lmem>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4x4xf32, #rair.space<lmem>>
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @move_size_mismatch(
    %src: memref<4xf32, #rair.space<host>>,
    %dst: memref<8xf32, #rair.space<host>>) {
  rair.scope @gemmini {
    %src_view = rair.view %src {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<4xf32, #rair.space<host>> -> !rair.region
    %dst_view = rair.view %dst {
      offsets = array<i64: 0>, sizes = array<i64: 8>,
      strides = array<i64: 1>
    } : memref<8xf32, #rair.space<host>> -> !rair.region
    // expected-error @+1 {{'rair.move' op requires equal positive static element counts}}
    rair.move %src_view to %dst_view : !rair.region, !rair.region
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @lease_not_released() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.reserve' op requires its lease to have exactly one use}}
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4x4xf32, #rair.space<spad>>, !rair.lease
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @buffer_use_after_release() {
  rair.scope @gemmini {
    // expected-error @+1 {{'rair.reserve' op reserved buffer is used after lease release}}
    %buffer, %lease = rair.reserve {space = #rair.space<spad>}
      : memref<4x4xf32, #rair.space<spad>>, !rair.lease
    rair.release_lease %lease, %buffer
      : !rair.lease, memref<4x4xf32, #rair.space<spad>>
    %view = rair.view %buffer {
      offsets = array<i64: 0, 0>, sizes = array<i64: 4, 4>,
      strides = array<i64: 1, 1>
    } : memref<4x4xf32, #rair.space<spad>> -> !rair.region
  }
  return
}

// -----

rair.target @gemmini {kind = "gemmini"}

func.func @compute_requires_matmul(
    %arg: memref<4xf32, #rair.space<host>>) {
  rair.scope @gemmini {
    %view = rair.view %arg {
      offsets = array<i64: 0>, sizes = array<i64: 4>,
      strides = array<i64: 1>
    } : memref<4xf32, #rair.space<host>> -> !rair.region
    // expected-error @+1 {{'rair.compute' op expects one static memref linalg.matmul payload in Core v0.1}}
    rair.compute ins(%view, %view : !rair.region, !rair.region)
      outs(%view : !rair.region) {
        memref.copy %arg, %arg
          : memref<4xf32, #rair.space<host>> to memref<4xf32, #rair.space<host>>
      }
  }
  return
}
