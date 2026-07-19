// RUN: torch-mlir-opt %s --rair-materialize-static-matmul | FileCheck %s --check-prefix=CORE
// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-materialize-static-matmul | FileCheck %s --check-prefix=IDEMPOTENT
// RUN: torch-mlir-opt %s --rair-materialize-static-matmul --rair-infer-effects -o /dev/null | FileCheck %s --check-prefix=REPORT

// CORE: rair.target @rair_default
// CORE-SAME: kind = "generic"
// CORE-LABEL: func.func @materialize_matmul
// CORE: rair.scope @rair_default
// CORE-COUNT-3: rair.view
// CORE-COUNT-3: rair.reserve
// CORE-COUNT-3: rair.view
// CORE-COUNT-3: rair.move
// CORE: rair.compute
// CORE: linalg.matmul
// CORE: rair.move
// CORE-COUNT-3: rair.release_lease

// IDEMPOTENT-COUNT-1: rair.target @rair_default
// IDEMPOTENT-COUNT-1: rair.scope @rair_default
// IDEMPOTENT-COUNT-1: rair.compute

// REPORT-LABEL: RAIR effect report: func @materialize_matmul scope 0 target @rair_default
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
// REPORT: summary: regions=6 actions=11

func.func @materialize_matmul(
    %a: memref<16x32xf32, #rair.space<host>>,
    %b: memref<32x8xf32, #rair.space<device>>,
    %c: memref<16x8xf32, #rair.space<host>>) {
  linalg.matmul
    ins(%a, %b
      : memref<16x32xf32, #rair.space<host>>,
        memref<32x8xf32, #rair.space<device>>)
    outs(%c : memref<16x8xf32, #rair.space<host>>)
  return
}
