//===- Ops.td - CIM dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the CIM dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_OPS
#define CIM_OPS


// include "torch-mlir/Dialect/CIM/IR/CIMBase.td"
// include "torch-mlir/Dialect/CIM/IR/CIMTypes.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

include "CIMDialect.td"

#endif
