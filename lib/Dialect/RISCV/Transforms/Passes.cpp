//===- Passes.cpp - RAIR transformation pass registration ----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RISCV/Transforms/RISCVPasses.h"

void rair::registerRAIRPasses() { rair::registerPasses(); }
