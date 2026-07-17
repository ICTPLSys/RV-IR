//===-- RISCVPasses.td - RAIR transformation passes -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES
#define TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES

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

#endif // TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES
