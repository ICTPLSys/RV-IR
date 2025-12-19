#ifndef CIM_DIALECT
#define CIM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CIM dialect definition.
//===----------------------------------------------------------------------===//

def CIM_Dialect : Dialect {
    // 定义名字空间 namespace，对应 C++ 的 getDialectNamespace 方法返回值
    let name = "cim";

    // 一行关于这个 Dialect 的介绍
    let summary = "CIM dialect.";

    // 更详细的关于这个 Dialect 的介绍
    let description = [{}];

    // 产生一个返回名字空间名称的接口
    let cppNamespace = "::cim";

    // 该设置用于激活 materializeConstant 方法，这使得可以例如 Canonicalize 优化
    // let hasConstantMaterializer = 1;

}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//
// 定义一个 Op 作为后续其他具体 Op 的基类
class CIM_Op<string mnemonic, list<Trait> traits = []> :
        Op<CIM_Dialect, mnemonic, traits>;

// Template for types.
class CIM_Type<string name, list<Trait> traits = []>
        : TypeDef<CIM_Dialect, name, traits>;
//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
// Define a type constraint that accepts all supported tensor types
// Using MLIR built-in tensor type definitions
def CIMTensor : AnyTypeOf<[
    I1Tensor, I8Tensor, I16Tensor, I32Tensor, I64Tensor,
    BF16Tensor, F16Tensor, F32Tensor, F64Tensor, AnySignlessIntegerOrIndex, AnyFloat
]>;

//Define sclar types
def CIMScalar : AnyTypeOf<[AnySignlessIntegerOrIndex,AnyFloat]>;

// Define integer tensor types
def CIMIntTensor : AnyTypeOf<[I8Tensor, I16Tensor, I32Tensor, I64Tensor]>;

// Define floating point tensor types  
def CIMFloatTensor : AnyTypeOf<[BF16Tensor, F16Tensor, F32Tensor, F64Tensor]>;






#endif // CIM_DIALECT