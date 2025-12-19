#ifndef RISCV_OPS
#define RISCV_OPS

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "RISCVDialect.td"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def ConstantOp : RISCV_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    Example:
    ```mlir
      %0 = "riscv.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>,
    OpBuilder<(ins "int64_t":$value)>
    
  ];
  // let results = (outs RISCVTensor);
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

}
//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
def PrintOp : RISCV_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
      The "print" builtin operation prints a given input tensor, and produces
      no results.
      
      Example:
      ```mlir
        "riscv.print"(%0) : (tensor<2x3xf64>) -> ()
       ```
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[RISCVTensor, AnyMemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

// RISC-V 可扩展 Accelerate ISA
//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
def MatmulOp : RISCV_Op<"matmul", [Pure]> {
    let summary = [{
      Performs a matrix multiplication of two 2D inputs without broadcast or transpose.
    }];
    let description = [{
      The "matmul" operation performs matrix multiplication between two
      tensors. The shapes of the tensor operands are expected to be compatible
      for matrix multiplication.
      
      Example MatMul:
      ```mlir
        %2 = "riscv.matmul"(%0, %1) : (tensor<2x3xf64>, tensor<3x4xf64>) -> tensor<2x4xf64>
      ```
    }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);

    // Indicate that the operation has a custom parser and printer method.
    // let hasCustomAssemblyFormat = 1;

    // Allow building a MatMulOp with from the two input operands.
    let builders = [
      OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}
//===----------------------------------------------------------------------===//
// MatvecOp
//===----------------------------------------------------------------------===//
def MatvecOp : RISCV_Op<"matvec", [Pure]> {
    let summary = [{
      Performs general matrix-vector multiplication (GEMV), computing the product 
      of a matrix and a vector with optional bias addition.
    }];
    let description = [{
      The "gemv" operation executes general matrix-vector multiplication, defined as:
      For inputs: matrix A (shape M×N), vector x (shape N), and optional bias vector b (shape M)
      Output: vector y (shape M), where y = Ax + b (if bias is present) or y = Ax (if no bias)
      The operation requires the matrix column dimension (N) to match the vector length (N) for
      valid multiplication.It supports floating-point and integer data types, and is typically
      mapped to RISC-V vector extensions or dedicated matrix accelerators.
      
      Example Gemv:
      ```mlir
        %2 = "riscv.matvec"(%0, %1) : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
      ```
    }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);

    let builders = [
      OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}
//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//
def ReduceOp : RISCV_Op<"reduce", [Pure]> {

  let summary = "tensor reduction operation";
  let description = [{
    Performs a reduction over one or more dimensions of a tensor.

    Example:
    ```mlir
      // Reduce add over dimension 0
        %res1 = "riscv.reduce"(%0) {kind = "sum",dim=[0]} : 
        (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>
      // Reduce add over dimension 1
        %res2 = "riscv.reduce"(%0) {kind = "sum",dim=[1]} : 
        (tensor<3x2x4xf64>) -> tensor<3x1x4xf64>
      // Reduce add over dimension 2
        %res3 = "riscv.reduce"(%0) {kind = "sum",dim=[2]} : 
        (tensor<3x2x4xf64>) -> tensor<3x2x1xf64>
      // Reduce add over dimension 0 and dimension 1
        %res4 = "riscv.reduce"(%0) {kind = "sum",dim=[0,1]} :
        (tensor<3x2x4xf64>) -> tensor<1x1x4xf64>

      // Reduce max over dimension 0 
        %res_max = "riscv.reduce"(%0) {kind = "max", dim=[0]} : (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>

      // Reduce min over dimension 0 
        %res_min = "riscv.reduce"(%0) {kind = "min", dim=[1]} : (tensor<3x2x4xf64>) -> tensor<3x1x4xf64>
    ```
  }];
  // let arguments = (ins RISCVTensor:$input, I64ArrayAttr:$axes, StrAttr:$kind);
  let arguments = (ins RISCVTensor:$input, I64ArrayAttr:$dim,StrAttr:$kind);
  let results = (outs RISCVTensor:$result);

    let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$input,"llvm::ArrayRef<int64_t>":$dim, "llvm::StringRef":$kind)>
  ];
  // let builders = [
  //   OpBuilder<(ins "mlir::Value":$input, "llvm::StringRef":$kind,
  //              CArg<"::mlir::arith::FastMathFlags",
  //                   "::mlir::arith::FastMathFlags::none">:$fastmath)>
  // ];
}
//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
def TransposeOp : RISCV_Op<"transpose", [Pure]> {
    // [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

  let summary = "transpose operation";
  let description = [{
    Example:
    ```mlir
      %1 = "riscv.transpose"(%0) : (tensor<2x3xf64>) -> (tensor<3x2xf64>)
      ```
  }];
  let arguments = (ins RISCVTensor:$input, I64ArrayAttr:$transp);
  // let arguments = (ins RISCVTensor:$input);

  let results = (outs RISCVTensor: $result);

    let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "mlir::Value":$input)>
  ];
  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : RISCV_Op<"reshape", [
    // DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
    %dest = "riscv.reshape"(%src, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$source,
    TensorRankOf<[AnySignlessInteger, Index], [1]>:$shape
  );
  let results = (outs AnyTensor:$result);

  let extraClassDeclaration = [{
    mlir::Type getResultType() { return getResult().getType(); }
  }];

  // let assemblyFormat = [{
  //   $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
  // }];

  // let hasVerifier = 1;

}
//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : RISCV_Op<"add", [Pure,Commutative]> {
  let summary = "General addition operation (supports integer/float)";
  let description = [{
    The `add` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<f64>)
    
    Examples:
    // Integer addition (scalar/Tensor)
    %int_scalar = "riscv.add"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.add"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);
    let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : RISCV_Op<"sub", [Pure]> {
  let summary = "General subtraction operation (supports integer/float)";
  let description = [{
    The `sub` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<f64>)
    
    Examples:
    // Integer subtraction (scalar/Tensor)
    %int_scalar = "riscv.sub"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.sub"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float subtraction (scalar/Tensor)
    %float_scalar = "riscv.sub"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "riscv.sub"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>

  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);

}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : RISCV_Op<"mul", [Pure]> {
  let summary = "General multiplication operation (supports integer/float)";
  let description = [{
    The `mul` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "riscv.mul"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.mul"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "riscv.mul"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "riscv.mul"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : RISCV_Op<"div", [Pure]> {
  let summary = "General division operation (supports signed/unsigned integer and float)";
  let description = [{
    The `div` operation takes two operands and returns one result, with all types consistent.
       Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "riscv.div"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.div"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "riscv.div"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "riscv.div"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def MaxOp : RISCV_Op<"max", [Pure]> {
  let summary = "General maximum operation (supports signed/unsigned integer and float)";
  let description = [{
    The `max` operation takes two operands and returns their maximum value, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "riscv.max"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.max"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "riscv.max"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "riscv.max"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);

}
//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def MinOp : RISCV_Op<"min", [Pure]> {
  let summary = "General minimum operation (supports signed/unsigned integer and float)";
  let description = [{
    The `min` operation takes two operands and returns their minimum value, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "riscv.min"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "riscv.min"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "riscv.min"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "riscv.min"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs);
    let results = (outs RISCVTensor);
}
//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def AndIOp : RISCV_Op<"andi",
    [Pure]> {
  let summary = "integer binary and";
  let description = [{
    The `andi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a tensor whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise and.
    %a = riscv.andi %b, %c : i64

    // Tensor element-wise bitwise integer and.
    %x = riscv.andi %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def XOrIOp : RISCV_Op<"xori",
    [Pure]> {
  let summary = "integer binary xor";
  let description = [{
    The `xori` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a tensor whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = riscv.xori %b, %c : i64

    // SIMD tensor element-wise bitwise integer xor.
    %f = riscv.xori %g, %h : tensor<4xi32>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def OrIOp : RISCV_Op<"ori",
    [Pure]> {
  let summary = "integer binary xor";
  let description = [{
    The `ori` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    tensor whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = riscv.ori %b, %c : i64

    // SIMD tensor element-wise bitwise integer xor.
    %f = riscv.ori %g, %h : tensor<4xi32>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def NegFOp : RISCV_Op<"negf",
    [Pure]> {
  let summary = "floating point negation";
  let description = [{
    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar negation value.
    %a = riscv.negf %b : f64

    // Tensor element-wise negation value.
    %x = riscv.negf %y : tensor<4x2xf8>
    ```
  }];

  let arguments = (ins FloatLike:$operand);
  let results = (outs FloatLike:$result);
}
//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//
def CmpIOp : RISCV_Op<"cmpi", [Pure]> {
  let summary = "integer comparison operation";
  let description = [{
    The `cmpi` operation is a generic comparison for integer-like types.  
    It compares two integer or tensor-of-integer operands according to the 
    predicate attribute, and produces an i1 or tensor-of-i1 result.

    Supported predicates (by string):
    - "eq"  : equal
    - "ne"  : not equal
    - "slt" : signed less than
    - "sle" : signed less or equal
    - "sgt" : signed greater than
    - "sge" : signed greater or equal
    - "ult" : unsigned less than
    - "ule" : unsigned less or equal
    - "ugt" : unsigned greater than
    - "uge" : unsigned greater or equal

    Example:

    ```mlir
    // Scalar comparison.
    %0 = "riscv.cmpi"(%lhs, %rhs) {predicate = "slt"} 
        : (i32, i32) -> i1

    // Tensor elementwise comparison.
    %1 = "riscv.cmpi"(%a, %b) {predicate = "eq"} 
        : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    ```
  }];

  // let arguments = (ins RISCVTensor:$lhs, RISCVTensor:$rhs, StrAttr:$predicate);
  // let arguments = (ins SignlessIntegerOrIndexLikeOfAnyRank:$lhs, SignlessIntegerOrIndexLikeOfAnyRank:$rhs, StrAttr:$predicate);
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs, StrAttr:$predicate);
  
  let results = (outs BoolLike:$result);
  // let results = (outs BoolLikeOfAnyRank:$result);


}
//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
def CmpFOp : RISCV_Op<"cmpf", [Pure]> {
  let summary = "floating-point comparison operation";
  let description = [{
    The `cmpf` operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.

    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type or tensor thereof.

    Example:

    ```mlir
    // Scalar comparison.
    %0 = "riscv.cmpf"(%lhs, %rhs) {predicate = "slt"} 
        : (f32, f32) -> i1
    //Tensor elementwise comparison.
    %1 = "riscv.cmpf"(%a, %b) {predicate = "eq"} 
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    ```
  }];


  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs, StrAttr:$predicate);

  let results = (outs BoolLike:$result);
  // let results = (outs BoolLikeOfAnyRank:$result);


}
//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------==
def BarrierOp : RISCV_Op<"barrier", []> {
    let summary = "Wait for dependent computations to finish.";
    let description = [{Blocks until the value of the associated future is present.
                       This operation can also be used to enforce a particular scheduling on the execution graph.}];

    let arguments = (ins RISCVTensor:$value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}
//===----------------------------------------------------------------------===//
// NocOp
//===----------------------------------------------------------------------==
def NocOp : RISCV_Op<"noc", []> {
    let summary = "On-Chip Network Communication.";
    let description = [{TODO.}];

    let arguments = (ins RISCVTensor:$value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : RISCV_Op<"conv2d", 
    [Pure]> {
    let summary = "RISCV dialect convolution 2d operation";
    let description = [{
        Performs a 2D convolution on the input tensor using specified kernel.
    }];

    let arguments = (ins RISCVTensor:$input, RISCVTensor:$kernel);
    // let arguments = (ins RISCVTensor:$input, RISCVTensor:$kernel, RISCVTensor:$bias);

    let results = (outs RISCVTensor:$output);

    // let builders = [
    //     OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$kernel, "mlir::Value":$bias)>
    // ];
    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$kernel)>
    ];
    // let extraClassDeclaration = [{
    //     static StringRef getStrideName() { return "stride"; }
    //     static StringRef getPaddingName() { return "padding"; }
    // }];

    // let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//
def LaunchOp : RISCV_Op<"launch", [  RISCV_AsyncOpInterface, 
                                     RISCV_HierarchyInterface, 
                                     AttrSizedOperandSegments,  //MLIR变长操作数的分段管理                                  
                                     IsolatedFromAbove,//MLIR特性，标记操作的region是上层隔离的
                                     AffineScope,  //MLIR特性，为操作创建一个仿射作用域
                                     SingleBlockImplicitTerminator<"LaunchTerminatorOp">]>,
                          Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                         Variadic<RISCV_AsyncToken>:$async_dependencies,
                                         Variadic<Index>:$sizes,
                                         Variadic<AnyType>:$launch_operands)>,                                        
                          Results<(outs Optional<RISCV_AsyncToken>:$async_token)>{
//async_token作为该launch操作完成的信号：当操作执行完毕后，会发出该令牌，后续依赖该操作的异步任务可通过此令牌同步
  let summary = "Launch";
  let description = [{Launch}];
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
   let extraClassDeclaration = [{
    ::mlir::ArrayRef<::mlir::BlockArgument> getIds();
    ::mlir::ArrayRef<::mlir::BlockArgument> getSize();
    ::mlir::OperandRange getSizeOperands();
    unsigned getNumKernelOperands();
    ::mlir::OperandRange getKernelOperands();
    ::mlir::Value getKernelOperand(unsigned i);
    ::mlir::ArrayRef<::mlir::BlockArgument> getKernelArguments();
    ::mlir::BlockArgument getKernelArgument(unsigned i);
    unsigned getNumDims();
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<::mlir::IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
    
    /// Return the kernel argument that corresponds to the given operand.
    /// Return an "empty" block argument if the given value is not a kernel
    /// operand.
    ::mlir::BlockArgument getTiedKernelArgument(::mlir::Value Oper) {
      auto kernelOperands = getKernelOperands();
      auto it = llvm::find(kernelOperands, Oper);
      if (it == kernelOperands.end())
        return {};
      return getKernelArgument(std::distance(kernelOperands.begin(), it));
    }
    /// Return the operand that corresponds to the given kernel argument.
    /// Return "nullptr" if the given block argument is not a kernel argument
    /// of this op.
    ::mlir::Value getTiedKernelOperand(::mlir::BlockArgument bbArg) {
      auto args = getKernelArguments();
      auto it = llvm::find(args, bbArg);
      if (it == args.end())
        return {};
      return getKernelOperand(std::distance(args.begin(), it));
    }

    /// Append the given values to kernel operands.
    void appendKernelOperands(::mlir::ValueRange operands){
      getLaunchOperandsMutable().append(operands);
      for (auto oper : operands){
        getBody().addArgument(oper.getType(), getLoc());
      }
    }
  }];
  // let skipDefaultBuilders = 1;
  // let hasCanonicalizer = 1;
}
//===----------------------------------------------------------------------===//
// LaunchTerminatorOp
//===----------------------------------------------------------------------===//
def LaunchTerminatorOp : RISCV_Op<"launch_terminator", [HasParent<"LaunchOp">,
                                                          Pure,
                                                          Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `riscv.launch`.";
  let description = [{
    A terminator operation for the body of `air.launch` operations.
    `air.launch` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}
//===----------------------------------------------------------------------===//
// HerdOp
//===----------------------------------------------------------------------===//
def HerdOp : RISCV_Op<"herd", [ RISCV_AsyncOpInterface, 
                                RISCV_HierarchyInterface, 
                                AttrSizedOperandSegments,                              
                                IsolatedFromAbove,
                                AffineScope,  
                                SingleBlockImplicitTerminator<"HerdTerminatorOp">]>,
                          Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                         OptionalAttr<StrAttr>:$link_with,
                                         Variadic<RISCV_AsyncToken>:$async_dependencies,
                                         Variadic<Index>:$sizes,
                                         Variadic<AnyType>:$herd_operands)>,                                        
                          Results<(outs Optional<RISCV_AsyncToken>:$async_token)>{
  let summary = "Herd";
  let description = [{Herd}];
  let regions = (region SizedRegion<1>:$body);
   let extraClassDeclaration = [{
    ::mlir::ArrayRef<::mlir::BlockArgument> getIds();
    ::mlir::ArrayRef<::mlir::BlockArgument> getSize();
    ::mlir::OperandRange getSizeOperands();
    unsigned getNumKernelOperands();
    ::mlir::OperandRange getKernelOperands();
    ::mlir::Value getKernelOperand(unsigned i);
    ::mlir::ArrayRef<::mlir::BlockArgument> getKernelArguments();
    ::mlir::BlockArgument getKernelArgument(unsigned i);
    unsigned getNumDims();
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<::mlir::IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
    static llvm::StringRef getColOffsetAttrName() { return "x_loc"; }
    static llvm::StringRef getRowOffsetAttrName() { return "y_loc"; }
    std::optional<uint64_t> getColOffset() {
      auto name = getColOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<::mlir::IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    std::optional<uint64_t> getRowOffset() {
      auto name = getRowOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<::mlir::IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    uint64_t getNumCols();
    uint64_t getNumRows();
    
    /// Return the kernel argument that corresponds to the given operand.
    /// Return an "empty" block argument if the given value is not a kernel
    /// operand.
    ::mlir::BlockArgument getTiedKernelArgument(::mlir::Value Oper) {
      auto kernelOperands = getKernelOperands();
      auto it = llvm::find(kernelOperands, Oper);
      if (it == kernelOperands.end())
        return {};
      return getKernelArgument(std::distance(kernelOperands.begin(), it));
    }
    /// Return the operand that corresponds to the given kernel argument.
    /// Return "nullptr" if the given block argument is not a kernel argument
    /// of this op.
    ::mlir::Value getTiedKernelOperand(::mlir::BlockArgument bbArg) {
      auto args = getKernelArguments();
      auto it = llvm::find(args, bbArg);
      if (it == args.end())
        return {};
      return getKernelOperand(std::distance(args.begin(), it));
    }

    /// Append the given values to kernel operands.
    void appendKernelOperands(::mlir::ValueRange operands){
      getHerdOperandsMutable().append(operands);
      for (auto oper : operands){
        getBody().addArgument(oper.getType(), getLoc());
      }
    }
  }];
  // let skipDefaultBuilders = 1;
  // let hasCanonicalizer = 1;
}
//===----------------------------------------------------------------------===//
// HerdTerminatorOp
//===----------------------------------------------------------------------===//
def HerdTerminatorOp : RISCV_Op<"herd_terminator", [HasParent<"HerdOp">,
                                                          Pure,
                                                          Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `riscv.herd`.";
  let description = [{
    A terminator operation for the body of `riscv.herd` operations.
    `riscv.herd` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}
//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//
def AllocOp: RISCV_Op<"alloc", []> {
  let summary = "RISCV Allocation Op";
  let arguments = (ins);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let description = [{
    AIRRt Allocation Op
  }];
  // let hasCanonicalizer  = 1;
}
//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//
def DeallocOp: RISCV_Op<"dealloc", []> {
  let summary = "RISCV Deallocation Op";
  let arguments = (ins AnyMemRef:$memref);

  let results = (outs);
  let assemblyFormat = [{
    attr-dict $memref `:` type($memref)
  }];
  let description = [{
    RISCV Deallocation Op
  }];
  // let hasCanonicalizer  = 1;
}
//===----------------------------------------------------------------------===//
// Template of *_DRV
//===----------------------------------------------------------------------===//
class RISCV_DRV_Op<string opName> : RISCV_Op<opName,
      [RISCV_AsyncOpInterface,
       RISCV_MemcpyInterface,
       AttrSizedOperandSegments]> {

  let summary = "dma operator";
  let arguments = (
    ins Variadic<RISCV_AsyncToken>:$async_dependencies,
        AnyRankedOrUnrankedMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyRankedOrUnrankedMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );

  let results = (outs Optional<RISCV_AsyncToken>:$async_token);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies)
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,`
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`
        attr-dict `:` `(` type($dst) `,` type($src) `)`
  }];

  let description = [{
    DMA operator
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getSrcMemref() { return getSrc(); }
    ::mlir::Value getDstMemref() { return getDst(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<::mlir::IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

//===----------------------------------------------------------------------===//
// *_DRV OP
//===----------------------------------------------------------------------===//
def LoadDrvOp : RISCV_DRV_Op<"load">;
def StoreDrvOp : RISCV_DRV_Op<"store">;


//===----------------------------------------------------------------------===//
// WaitAllOp
//===----------------------------------------------------------------------===//
def WaitAllOp: RISCV_Op<"wait_all", [RISCV_AsyncOpInterface]> {
  let arguments = (ins Variadic<RISCV_AsyncToken>:$async_dependencies);
  let results = (
    outs Optional<RISCV_AsyncToken>:$async_token
  );

  let summary = "wait for all operator";
  let description = [{
    Wait for all async tokens before preceding.
    Example:
        %0 = air.wait_all async 
        %4 = riscv.wait_all async [%a4, %a5] 
  }];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies) attr-dict
  }];
  // let extraClassDeclaration = [{
  //   int32_t getId() {
  //     if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
  //       return id_attr.getInt();
  //     }
  //     return -1;
  //   }
  // }];
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// WorldOp
//===----------------------------------------------------------------------===//
def WorldOp : RISCV_Op<"world", [Pure]> {
    let summary = "print RISCV, World";
    let description = [{
        The "world" operation prints "RISCV, World", and produces
        no results.
    }];
}


#endif // RISCV_OPS
