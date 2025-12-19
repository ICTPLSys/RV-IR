#ifndef RISCV_DIALECT
#define RISCV_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// RISCV dialect definition.
//===----------------------------------------------------------------------===//

def RISCV_Dialect : Dialect {
    // 定义名字空间 namespace，对应 C++ 的 getDialectNamespace 方法返回值
    let name = "riscv";

    // 一行关于这个 Dialect 的介绍
    let summary = "RISC-V dialect.";

    // 更详细的关于这个 Dialect 的介绍
    let description = [{
        RISC-V Universal ISA + RISC-V Scalable Accelerate ISA.
    }];

    // 产生一个返回名字空间名称的接口
    let cppNamespace = "::riscv";

    // 该设置用于激活 materializeConstant 方法，这使得可以例如 Canonicalize 优化
    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;

}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//
// 定义一个 Op 作为后续其他具体 Op 的基类
class RISCV_Op<string mnemonic, list<Trait> traits = []> :
        Op<RISCV_Dialect, mnemonic, traits>;

// Template for types.
class RISCV_Type<string name, list<Trait> traits = []>
        : TypeDef<RISCV_Dialect, name, traits>;
//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
// Define a type constraint that accepts all supported tensor types
// Using MLIR built-in tensor type definitions
def RISCVTensor : AnyTypeOf<[
    I1Tensor, I8Tensor, I16Tensor, I32Tensor, I64Tensor,
    BF16Tensor, F16Tensor, F32Tensor, F64Tensor, AnySignlessIntegerOrIndex, AnyFloat
]>;

//Define sclar types
def RISCVScalar : AnyTypeOf<[AnySignlessIntegerOrIndex,AnyFloat]>;

// Define integer tensor types
def RISCVIntTensor : AnyTypeOf<[I8Tensor, I16Tensor, I32Tensor, I64Tensor]>;

// Define floating point tensor types  
def RISCVFloatTensor : AnyTypeOf<[BF16Tensor, F16Tensor, F32Tensor, F64Tensor]>;

//===----------------------------------------------------------------------===//
// Interface definitions
//===----------------------------------------------------------------------===//

def RISCV_AsyncToken : DialectType<
  RISCV_Dialect, CPred<"llvm::isa<::riscv::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"::riscv::AsyncTokenType::get($_builder.getContext())">;

def RISCV_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for RISCV operations that execute asynchronously.

    RISCV operations implementing this interface take a list of dependencies
    as `riscv.async.token` arguments and optionally return a `riscv.async.token`.

    The op doesn't start executing until all ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op is scheduled for execution on the
    device and returns immediately, without waiting for the execution to
    complete. If the op does not return a token, the op will wait for
    execution to complete.
  }];
  let cppNamespace = "::riscv";
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
def RISCV_HierarchyInterface : OpInterface<"HierarchyInterface"> {
  let description = [{
    Interface for riscv hierarchy ops `riscv.herd`, `riscv.segment` and `riscv.launch`

    Operations implementing this interface define an iteration space, support
    arguments for implementing "isolated from above" and have an integerc `id`
    attribute.
  }];
  let cppNamespace = "::riscv";
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

def RISCV_MemcpyInterface : OpInterface<"MemcpyInterface"> {
  let description = [{}];
  let cppNamespace = "::riscv";
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
def RISCV_Event : DialectType<
  RISCV_Dialect, CPred<"llvm::isa<::riscv::EventType>($_self)">, "event type">,
             BuildableType<"::riscv::EventType::get($_builder.getContext())">;

def GMEM:  I32EnumAttrCase<"GMEM", 6>;
def SPAD0:  I32EnumAttrCase<"SPAD0", 5>;
def SPAD1:  I32EnumAttrCase<"SPAD1", 4>;
def SPAD2:  I32EnumAttrCase<"SPAD2", 3>;
def SPAD3:  I32EnumAttrCase<"SPAD3", 2>;
def CIMC0:  I32EnumAttrCase<"CIMC0", 1>;
def LMEM:  I32EnumAttrCase<"LMEM", 0>;

def MemorySpace: I32EnumAttr<"MemorySpace", "AIR Memory Space IDs",
  [GMEM,SPAD0,SPAD1,SPAD2,SPAD3,CIMC0,LMEM]> {

  let cppNamespace = "::riscv";
}





#endif // RISCV_DIALECT