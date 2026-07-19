//===- RAIRPasses.h - RAIR transformation passes --------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES_H
#define TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace rair {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createVerifyLifetimesPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createInferEffectsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMaterializeStaticMatmulPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMaterializePlanPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createVerifyPlanPass();

/// Registers all RAIR transformation and verification passes.
void registerRAIRPasses();

#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h.inc"

} // namespace rair

#endif // TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPASSES_H
