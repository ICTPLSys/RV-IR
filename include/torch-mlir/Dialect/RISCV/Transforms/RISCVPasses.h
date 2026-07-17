//===- RISCVPasses.h - RAIR transformation passes --------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES_H
#define TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace rair {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createVerifyLifetimesPass();

/// Registers all RAIR transformation and verification passes.
void registerRAIRPasses();

#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/RISCV/Transforms/RISCVPasses.h.inc"

} // namespace rair

#endif // TORCHMLIR_DIALECT_RISCV_TRANSFORMS_RISCVPASSES_H
