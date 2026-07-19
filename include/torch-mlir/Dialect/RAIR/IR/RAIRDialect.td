#ifndef RAIR_DIALECT
#define RAIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// RAIR dialect definition.
//===----------------------------------------------------------------------===//

def RAIR_Dialect : Dialect {
    // 定义名字空间 namespace，对应 C++ 的 getDialectNamespace 方法返回值
    let name = "rair";

    // 一行关于这个 Dialect 的介绍
    let summary = "Resource-Adaptive Intermediate Representation dialect.";

    // 更详细的关于这个 Dialect 的介绍
    let description = [{
        RAIR models target-neutral resource ownership, memory regions, data
        movement, compute contracts, and effect-preserving execution plans.
    }];

    // 产生一个返回名字空间名称的接口
    let cppNamespace = "::rair";

    // 该设置用于激活 materializeConstant 方法，这使得可以例如 Canonicalize 优化
    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//
// 定义一个 Op 作为后续其他具体 Op 的基类
class RAIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<RAIR_Dialect, mnemonic, traits>;

// Template for types.
class RAIR_Type<string name, list<Trait> traits = []>
        : TypeDef<RAIR_Dialect, name, traits>;
//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

def RAIR_AsyncTokenType : RAIR_Type<"AsyncToken"> {
  let mnemonic = "async.token";
  let summary = "legacy asynchronous completion token";
  let assemblyFormat = "";
}

def RAIR_EventType : RAIR_Type<"Event"> {
  let mnemonic = "event";
  let summary = "target-aware RAIR Plan completion event";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$target);
  let assemblyFormat = "`<` $target `>`";
}

def RAIR_ContextType : RAIR_Type<"Context"> {
  let mnemonic = "context";
  let summary = "legacy accelerator execution context";
  let assemblyFormat = "";
}

def RAIR_RegionType : RAIR_Type<"Region"> {
  let mnemonic = "region";
  let summary = "analyzable static buffer region";
  let assemblyFormat = "";
}

def RAIR_LeaseType : RAIR_Type<"Lease"> {
  let mnemonic = "lease";
  let summary = "linear reservation lifetime capability";
  let assemblyFormat = "";
}

// Define a type constraint that accepts all supported tensor types
// Using MLIR built-in tensor type definitions
def RAIRTensor : AnyTypeOf<[
    I1Tensor, I8Tensor, I16Tensor, I32Tensor, I64Tensor,
    BF16Tensor, F16Tensor, F32Tensor, F64Tensor, AnySignlessIntegerOrIndex, AnyFloat
]>;

//Define sclar types
def RAIRScalar : AnyTypeOf<[AnySignlessIntegerOrIndex,AnyFloat]>;

// Define integer tensor types
def RAIRIntTensor : AnyTypeOf<[I8Tensor, I16Tensor, I32Tensor, I64Tensor]>;

// Define floating point tensor types  
def RAIRFloatTensor : AnyTypeOf<[BF16Tensor, F16Tensor, F32Tensor, F64Tensor]>;

//===----------------------------------------------------------------------===//
// Interface definitions
//===----------------------------------------------------------------------===//

def RAIR_AsyncToken : DialectType<
  RAIR_Dialect, CPred<"llvm::isa<::rair::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"::rair::AsyncTokenType::get($_builder.getContext())">;

def RAIR_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for RAIR operations that execute asynchronously.

    RAIR operations implementing this interface take a list of dependencies
    as `rair.async.token` arguments and optionally return a `rair.async.token`.

    The op doesn't start executing until all ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op is scheduled for execution on the
    device and returns immediately, without waiting for the execution to
    complete. If the op does not return a token, the op will wait for
    execution to complete.
  }];
  let cppNamespace = "::rair";
  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "::mlir::OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "::mlir::Value", "getAsyncToken"
    >
  ];
}
def RAIR_HierarchyInterface : OpInterface<"HierarchyInterface"> {
  let description = [{
    Interface for RAIR hierarchy ops `rair.herd`, `rair.segment` and `rair.launch`

    Operations implementing this interface define an iteration space, support
    arguments for implementing "isolated from above" and have an integerc `id`
    attribute.
  }];
  let cppNamespace = "::rair";
  let methods = [
    InterfaceMethod<"Get the block arguments representing the iteration space indices (e.g., %x, %y for a 2D herd)",
    "::mlir::ArrayRef<::mlir::BlockArgument>", "getIds"
    >,
    InterfaceMethod<"Get the block arguments representing the iteration space sizes (e.g., %sx, %sy for a 2D herd)",
    "::mlir::ArrayRef<::mlir::BlockArgument>", "getSize"
    >,
    InterfaceMethod<"Get the operands that define the iteration space sizes",
    "::mlir::OperandRange", "getSizeOperands"
    >,
    InterfaceMethod<"Get the number of kernel operands passed to the hierarchy operation",
    "unsigned", "getNumKernelOperands"
    >,
    InterfaceMethod<"Get all block arguments representing kernel operands (arguments passed from outside)",
    "::mlir::ArrayRef<::mlir::BlockArgument>", "getKernelArguments"
    >,
    InterfaceMethod<"Get all operands passed as kernel arguments to the hierarchy operation",
    "::mlir::OperandRange", "getKernelOperands"
    >,
    InterfaceMethod<"Get the i-th kernel operand",
    "::mlir::Value", "getKernelOperand", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Get the i-th kernel argument (block argument corresponding to kernel operand)",
    "::mlir::BlockArgument", "getKernelArgument", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Get the number of dimensions in the iteration space",
    "unsigned", "getNumDims"
    >,
    InterfaceMethod<"Get the unique identifier for this hierarchy operation",
    "int32_t", "getId"
    >,
    InterfaceMethod<"Get the block argument that corresponds to the given kernel operand",
    "::mlir::BlockArgument", "getTiedKernelArgument", (ins "::mlir::Value":$i)
    >,
    InterfaceMethod<"Get the kernel operand that corresponds to the given block argument",
    "::mlir::Value", "getTiedKernelOperand", (ins "::mlir::BlockArgument":$i)
    >,
    InterfaceMethod<"Get the body region of the hierarchy operation",
    "::mlir::Region&", "getBody"
    >,
    InterfaceMethod<"Append new kernel operands to the hierarchy operation",
    "void", "appendKernelOperands", (ins "::mlir::ValueRange":$i)
    >,
  ];
}

def RAIR_MemcpyInterface : OpInterface<"MemcpyInterface"> {
  let description = [{}];
  let cppNamespace = "::rair";
  let methods = [
    InterfaceMethod<"description",
    "::mlir::Value", "getSrcMemref"
    >,
    InterfaceMethod<"description",
    "::mlir::Value", "getDstMemref"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getSrcOffsets"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getDstOffsets"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getSrcSizes"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getDstSizes"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getSrcStrides"
    >,
    InterfaceMethod<"",
    "::mlir::OperandRange", "getDstStrides"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}
//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
def RAIR_Event : DialectType<
  RAIR_Dialect, CPred<"llvm::isa<::rair::EventType>($_self)">, "event type">;

def RAIR_Context : DialectType<
  RAIR_Dialect, CPred<"llvm::isa<::rair::ContextType>($_self)">, "accelerator context type">,
             BuildableType<"::rair::ContextType::get($_builder.getContext())">;

// Values 0-6 are kept temporarily for the active pre-Core lowering. New Core
// IR must use the target-neutral values below and may not use `unknown`.
def GMEM:  I32EnumAttrCase<"GMEM", 6, "gmem">;
def SPAD0:  I32EnumAttrCase<"SPAD0", 5, "spad0">;
def SPAD1:  I32EnumAttrCase<"SPAD1", 4, "spad1">;
def SPAD2:  I32EnumAttrCase<"SPAD2", 3, "spad2">;
def SPAD3:  I32EnumAttrCase<"SPAD3", 2, "spad3">;
def CIMC0:  I32EnumAttrCase<"CIMC0", 1, "cimc0">;
def LMEM:  I32EnumAttrCase<"LMEM", 0, "lmem">;
def HOST: I32EnumAttrCase<"HOST", 7, "host">;
def DEVICE: I32EnumAttrCase<"DEVICE", 8, "device">;
def SPAD: I32EnumAttrCase<"SPAD", 9, "spad">;
def ACC: I32EnumAttrCase<"ACC", 10, "acc">;
def UNKNOWN: I32EnumAttrCase<"UNKNOWN", 11, "unknown">;

def MemorySpace: I32EnumAttr<"MemorySpace", "RAIR memory space kinds",
  [GMEM,SPAD0,SPAD1,SPAD2,SPAD3,CIMC0,LMEM,
   HOST,DEVICE,SPAD,ACC,UNKNOWN]> {

  let cppNamespace = "::rair";
  let genSpecializedAttr = 0;
}

def RAIR_MemorySpaceAttr :
    EnumAttr<RAIR_Dialect, MemorySpace, "space"> {
  let summary = "RAIR memory space";
  let description = [{
    Identifies the memory domain used by a RAIR buffer or transfer endpoint.
    Core IR uses target-neutral `host`, `device`, `spad`, or `acc` values.
    The older `gmem`, `lmem`, numbered scratchpad, and CIM values remain only
    while the pre-Core Linalg lowering is migrated.
  }];
  let assemblyFormat = "`<` $value `>`";
}

def READ: I32EnumAttrCase<"Read", 0, "read">;
def WRITE: I32EnumAttrCase<"Write", 1, "write">;
def READWRITE: I32EnumAttrCase<"ReadWrite", 2, "readwrite">;

def AccessMode : I32EnumAttr<"AccessMode", "RAIR compute access mode",
  [READ, WRITE, READWRITE]> {
  let cppNamespace = "::rair";
  let genSpecializedAttr = 0;
}

def RAIR_AccessModeAttr : EnumAttr<RAIR_Dialect, AccessMode, "access"> {
  let summary = "RAIR compute operand access contract";
  let assemblyFormat = "`<` $value `>`";
}

def TASK_RESERVE: I32EnumAttrCase<"Reserve", 0, "reserve">;
def TASK_MOVE: I32EnumAttrCase<"Move", 1, "move">;
def TASK_COMPUTE: I32EnumAttrCase<"Compute", 2, "compute">;
def TASK_RELEASE_LEASE:
    I32EnumAttrCase<"ReleaseLease", 3, "release_lease">;

def TaskKind : I32EnumAttr<"TaskKind", "RAIR Plan source action kind",
  [TASK_RESERVE, TASK_MOVE, TASK_COMPUTE, TASK_RELEASE_LEASE]> {
  let cppNamespace = "::rair";
  let genSpecializedAttr = 0;
}

def RAIR_TaskKindAttr : EnumAttr<RAIR_Dialect, TaskKind, "task_kind"> {
  let summary = "kind of the Core action represented by a Plan task";
  let assemblyFormat = "`<` $value `>`";
}



#endif // RAIR_DIALECT
