// RUN: torch-mlir-opt %s --canonicalize --cse | FileCheck %s
// RUN: torch-mlir-opt %s --rair-infer-effects -o /dev/null | FileCheck %s --check-prefix=REPORT
// RUN: torch-mlir-opt %s --rair-infer-effects | FileCheck %s --check-prefix=PRESERVE

// CHECK: rair.target @gemmini
// CHECK-LABEL: func.func @core_static_matmul
// CHECK: rair.scope @gemmini
// CHECK: %[[A_SPAD:.*]], %[[A_LEASE:.*]] = rair.reserve
// CHECK: %[[B_SPAD:.*]], %[[B_LEASE:.*]] = rair.reserve
// CHECK: %[[C_ACC:.*]], %[[C_LEASE:.*]] = rair.reserve
// CHECK: rair.move %{{.*}} to %{{.*}}
// CHECK: rair.move %{{.*}} to %{{.*}}
// CHECK: rair.move %{{.*}} to %{{.*}}
// CHECK: rair.compute
// CHECK: linalg.matmul
// CHECK: rair.move %{{.*}} to %{{.*}}
// CHECK: rair.release_lease %[[C_LEASE]], %[[C_ACC]]
// CHECK: rair.release_lease %[[B_LEASE]], %[[B_SPAD]]
// CHECK: rair.release_lease %[[A_LEASE]], %[[A_SPAD]]

// REPORT-LABEL: RAIR effect report: func @core_static_matmul scope 0 target @gemmini
// REPORT: r0 base=arg0 space=host offsets=[0, 0] sizes=[16, 32] strides=[1, 1]
// REPORT: r3 base=a0.buffer space=spad offsets=[0, 0] sizes=[16, 32] strides=[1, 1]
// REPORT: a3 move effects=[read(r0), write(r3)]
// REPORT: a4 move effects=[read(r1), write(r4)]
// REPORT: a5 move effects=[read(r2), write(r5)]
// REPORT: a6 compute effects=[read(r3), read(r4), readwrite(r5)]
// REPORT: a7 move effects=[read(r5), write(r2)]
// REPORT: a3 -> a6 memory conflict=RAW write(r3) -> read(r3) relation=overlap
// REPORT: a4 -> a6 memory conflict=RAW write(r4) -> read(r4) relation=overlap
// REPORT: a5 -> a6 memory conflict=RAW+WAW write(r5) -> readwrite(r5) relation=overlap
// REPORT: a6 -> a7 memory conflict=RAW readwrite(r5) -> read(r5) relation=overlap
// REPORT: independent:
// REPORT: a3 || a4

// PRESERVE: RAIR effect report: func @core_static_matmul
// PRESERVE: module {
// PRESERVE: rair.target @gemmini
// PRESERVE: func.func @core_static_matmul
// PRESERVE: rair.scope @gemmini
// PRESERVE: rair.compute

rair.target @gemmini {
  kind = "gemmini",
  spad_bytes = 262144 : i64,
  acc_bytes = 65536 : i64
}

func.func private @event_type_is_target_aware(!rair.event<@gemmini>)

func.func @core_static_matmul(
    %a: memref<16x32xf32, #rair.space<host>>,
    %b: memref<32x16xf32, #rair.space<host>>,
    %c: memref<16x16xf32, #rair.space<host>>) {
  rair.scope @gemmini {
    %a_host = rair.view %a {
      offsets = array<i64: 0, 0>, sizes = array<i64: 16, 32>,
      strides = array<i64: 1, 1>
    } : memref<16x32xf32, #rair.space<host>> -> !rair.region
    %b_host = rair.view %b {
      offsets = array<i64: 0, 0>, sizes = array<i64: 32, 16>,
      strides = array<i64: 1, 1>
    } : memref<32x16xf32, #rair.space<host>> -> !rair.region
    %c_host = rair.view %c {
      offsets = array<i64: 0, 0>, sizes = array<i64: 16, 16>,
      strides = array<i64: 1, 1>
    } : memref<16x16xf32, #rair.space<host>> -> !rair.region

    %a_spad, %a_lease = rair.reserve {space = #rair.space<spad>}
      : memref<16x32xf32, #rair.space<spad>>, !rair.lease
    %b_spad, %b_lease = rair.reserve {space = #rair.space<spad>}
      : memref<32x16xf32, #rair.space<spad>>, !rair.lease
    %c_acc, %c_lease = rair.reserve {space = #rair.space<acc>}
      : memref<16x16xf32, #rair.space<acc>>, !rair.lease

    %a_local = rair.view %a_spad {
      offsets = array<i64: 0, 0>, sizes = array<i64: 16, 32>,
      strides = array<i64: 1, 1>
    } : memref<16x32xf32, #rair.space<spad>> -> !rair.region
    %b_local = rair.view %b_spad {
      offsets = array<i64: 0, 0>, sizes = array<i64: 32, 16>,
      strides = array<i64: 1, 1>
    } : memref<32x16xf32, #rair.space<spad>> -> !rair.region
    %c_local = rair.view %c_acc {
      offsets = array<i64: 0, 0>, sizes = array<i64: 16, 16>,
      strides = array<i64: 1, 1>
    } : memref<16x16xf32, #rair.space<acc>> -> !rair.region

    rair.move %a_host to %a_local : !rair.region, !rair.region
    rair.move %b_host to %b_local : !rair.region, !rair.region
    rair.move %c_host to %c_local : !rair.region, !rair.region
    rair.compute
      ins(%a_local, %b_local : !rair.region, !rair.region)
      outs(%c_local : !rair.region) {
        linalg.matmul
          ins(%a_spad, %b_spad
            : memref<16x32xf32, #rair.space<spad>>,
              memref<32x16xf32, #rair.space<spad>>)
          outs(%c_acc : memref<16x16xf32, #rair.space<acc>>)
      }
    rair.move %c_local to %c_host : !rair.region, !rair.region

    rair.release_lease %c_lease, %c_acc
      : !rair.lease, memref<16x16xf32, #rair.space<acc>>
    rair.release_lease %b_lease, %b_spad
      : !rair.lease, memref<32x16xf32, #rair.space<spad>>
    rair.release_lease %a_lease, %a_spad
      : !rair.lease, memref<16x32xf32, #rair.space<spad>>
  }
  return
}
