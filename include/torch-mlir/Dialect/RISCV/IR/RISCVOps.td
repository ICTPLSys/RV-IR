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
def ConstantOp : RAIR_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    Example:
    ```mlir
      %0 = "rair.constant"()
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
    // OpBuilder<(ins "int":$value)>
  ];
  // let results = (outs RAIRTensor);
  // let arguments = (ins ElementsAttr:$value);
  // let results = (outs RAIRTensor);
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

}
//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
def PrintOp : RAIR_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
      The "print" builtin operation prints a given input tensor, and produces
      no results.
      
      Example:
      ```mlir
        "rair.print"(%0) : (tensor<2x3xf64>) -> ()
       ```
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[RAIRTensor, AnyMemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}


//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
def MatmulOp : RAIR_Op<"matmul", [Pure]> {
    let summary = [{
      Performs a matrix multiplication of two 2D inputs without broadcast or transpose.
    }];
    let description = [{
      The "matmul" operation performs matrix multiplication between two
      tensors. The shapes of the tensor operands are expected to be compatible
      for matrix multiplication.

      Example MatMul:
      ```mlir
      rair.matmul ins(%lhs, %rhs : memref<2x3xf64>, memref<3x4xf64s>)
                   outs(%out : memref<2x4xf64>)
                   {accelerator = "gemmini", tile_size = array<i64: 16, 16, 16>}
      ```
    }];

    let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output,
                         OptionalAttr<StrAttr>:$accelerator,
                         OptionalAttr<DenseI64ArrayAttr>:$tile_size,
                         OptionalAttr<StrAttr>:$dataflow);
    let results = (outs);

    // Allow building a MatMulOp with from the two input operands.
    let builders = [
      OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
    let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict
    }];

}
// RISC-V 可扩展 Accelerate ISA
//===----------------------------------------------------------------------===//
// BatchMatMulOp
//===----------------------------------------------------------------------===//
def BatchMatMulOp : RAIR_Op<"batch_matmul", [Pure]> {
  let summary = [{
    Performs an in-place batch matrix multiplication.
  }];

  let description = [{
    The "batchmatmul" operation performs batch matrix multiplication:
    for each batch b:
      output[b] = lhs[b] * rhs[b]

    All operands are memrefs. The result is written into `output`.

    Example:
    ```mlir
    rair.batch_matmul ins(%lhs, %rhs : memref<1x128x2048xf32>, memref<1x2048x8192xf32>)
                      outs(%out : memref<1x128x8192xf32>)
                      {accelerator = "gemmini"}
    ```
  }];

  // In-place semantics: no results
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$output,
                       OptionalAttr<StrAttr>:$accelerator,
                       OptionalAttr<DenseI64ArrayAttr>:$tile_size,
                       OptionalAttr<StrAttr>:$dataflow);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// MatvecOp
//===----------------------------------------------------------------------===//
def MatvecOp : RAIR_Op<"matvec", [Pure]> {
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
        %2 = "rair.matvec"(%0, %1) : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
      ```
    }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator,
                         OptionalAttr<DenseI64ArrayAttr>:$tile_size);
    let results = (outs RAIRTensor);

    let builders = [
      OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}
//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//
def ReduceOp : RAIR_Op<"reduce", [Pure]> {

  let summary = "tensor reduction operation";
  let description = [{
    Performs a reduction over one or more dimensions of a tensor.

    Example:
    ```mlir
      // Reduce add over dimension 0
        %res1 = "rair.reduce"(%0) {kind = "sum",dim=[0]} : 
        (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>
      // Reduce add over dimension 1
        %res2 = "rair.reduce"(%0) {kind = "sum",dim=[1]} : 
        (tensor<3x2x4xf64>) -> tensor<3x1x4xf64>
      // Reduce add over dimension 2
        %res3 = "rair.reduce"(%0) {kind = "sum",dim=[2]} : 
        (tensor<3x2x4xf64>) -> tensor<3x2x1xf64>
      // Reduce add over dimension 0 and dimension 1
        %res4 = "rair.reduce"(%0) {kind = "sum",dim=[0,1]} :
        (tensor<3x2x4xf64>) -> tensor<1x1x4xf64>

      // Reduce max over dimension 0 
        %res_max = "rair.reduce"(%0) {kind = "max", dim=[0]} : (tensor<3x2x4xf64>) -> tensor<1x2x4xf64>

      // Reduce min over dimension 0 
        %res_min = "rair.reduce"(%0) {kind = "min", dim=[1]} : (tensor<3x2x4xf64>) -> tensor<3x1x4xf64>
    ```
  }];
  let arguments = (ins RAIRTensor:$input, I64ArrayAttr:$dim, StrAttr:$kind,
                       OptionalAttr<StrAttr>:$accelerator);
  let results = (outs RAIRTensor:$result);

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

def TransposeOp : RAIR_Op<"transpose", [Pure]> {
    // [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

  let summary = "transpose operation";
  let description = [{
    Example:
    ```mlir
      rair.transpose ins(%0 : memref<64x32xf32>) outs(%alloc : memref<32x64xf32>) {permutation = array<i64: 1, 0>}
      ```
  }];
  let arguments = (ins
    AnyMemRef :$input,
    AnyMemRef :$init,
    DenseI64ArrayAttr:$permutation,  // 转置维度
    OptionalAttr<StrAttr>:$accelerator
  );

  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `)` `outs` `(` $init `:` type($init) `)` attr-dict
  }];

}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : RAIR_Op<"reshape", [
    // DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
    %dest = "rair.reshape"(%src, %shape) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
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

}
//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : RAIR_Op<"add", [Pure,Commutative]> {
  let summary = "General addition operation (supports integer/float)";
  let description = [{
    The `add` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<f64>)

    Examples:
    // Integer addition (scalar/Tensor)
    %int_scalar = "rair.add"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.add"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);
    let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : RAIR_Op<"sub", [Pure]> {
  let summary = "General subtraction operation (supports integer/float)";
  let description = [{
    The `sub` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<f64>)
    
    Examples:
    // Integer subtraction (scalar/Tensor)
    %int_scalar = "rair.sub"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.sub"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float subtraction (scalar/Tensor)
    %float_scalar = "rair.sub"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "rair.sub"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>

  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);

}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : RAIR_Op<"mul", [Pure]> {
  let summary = "General multiplication operation (supports integer/float)";
  let description = [{
    The `mul` operation takes two operands and returns one result, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "rair.mul"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.mul"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "rair.mul"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "rair.mul"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : RAIR_Op<"div", [Pure]> {
  let summary = "General division operation (supports signed/unsigned integer and float)";
  let description = [{
    The `div` operation takes two operands and returns one result, with all types consistent.
       Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "rair.div"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.div"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "rair.div"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "rair.div"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def MaxOp : RAIR_Op<"max", [Pure]> {
  let summary = "General maximum operation (supports signed/unsigned integer and float)";
  let description = [{
    The `max` operation takes two operands and returns their maximum value, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "rair.max"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.max"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "rair.max"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "rair.max"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);

}
//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def MinOp : RAIR_Op<"min", [Pure]> {
  let summary = "General minimum operation (supports signed/unsigned integer and float)";
  let description = [{
    The `min` operation takes two operands and returns their minimum value, with all types consistent.
    Supported types:
    - Integer: Signless integer scalar/Tensor (e.g., i32, tensor<4xi64>)
    - Float: Floating-point scalar/Tensor (e.g., f16, tensor<2xf32>, tensor<f64>)
    
    Examples:
    // Integer multiplication (scalar/Tensor)
    %int_scalar = "rair.min"(%a, %b) : (i32, i32) -> i32
    %int_tensor = "rair.min"(%t_a, %t_b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>

    // Float multiplication (scalar/Tensor)
    %float_scalar = "rair.min"(%c, %d) : (f32, f32) -> f32
    %float_tensor = "rair.min"(%vec_c, %vec_d) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  }];

    let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs,
                         OptionalAttr<StrAttr>:$accelerator);
    let results = (outs RAIRTensor);
}
//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def AndIOp : RAIR_Op<"andi",
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
    %a = rair.andi %b, %c : i64

    // Tensor element-wise bitwise integer and.
    %x = rair.andi %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def XOrIOp : RAIR_Op<"xori",
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
    %a = rair.xori %b, %c : i64

    // SIMD tensor element-wise bitwise integer xor.
    %f = rair.xori %g, %h : tensor<4xi32>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// OrIOp   
//===----------------------------------------------------------------------===//

def OrIOp : RAIR_Op<"ori",
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
    %a = rair.ori %b, %c : i64

    // SIMD tensor element-wise bitwise integer xor.
    %f = rair.ori %g, %h : tensor<4xi32>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);
}
//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def NegFOp : RAIR_Op<"negf",
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
    %a = rair.negf %b : f64

    // Tensor element-wise negation value.
    %x = rair.negf %y : tensor<4x2xf8>
    ```
  }];

  let arguments = (ins FloatLike:$operand,
                       OptionalAttr<StrAttr>:$accelerator);
  let results = (outs FloatLike:$result);
}
//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//
def CmpIOp : RAIR_Op<"cmpi", [Pure]> {
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
    %0 = "rair.cmpi"(%lhs, %rhs) {predicate = "slt"} 
        : (i32, i32) -> i1

    // Tensor elementwise comparison.
    %1 = "rair.cmpi"(%a, %b) {predicate = "eq"} 
        : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    ```
  }];

  // let arguments = (ins RAIRTensor:$lhs, RAIRTensor:$rhs, StrAttr:$predicate);
  // let arguments = (ins SignlessIntegerOrIndexLikeOfAnyRank:$lhs, SignlessIntegerOrIndexLikeOfAnyRank:$rhs, StrAttr:$predicate);
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs, StrAttr:$predicate);
  
  let results = (outs BoolLike:$result);
  // let results = (outs BoolLikeOfAnyRank:$result);


}
//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
def CmpFOp : RAIR_Op<"cmpf", [Pure]> {
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
    %0 = "rair.cmpf"(%lhs, %rhs) {predicate = "slt"} 
        : (f32, f32) -> i1
    //Tensor elementwise comparison.
    %1 = "rair.cmpf"(%a, %b) {predicate = "eq"} 
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
def BarrierOp : RAIR_Op<"barrier", []> {
    let summary = "Wait for dependent computations to finish.";
    let description = [{Blocks until the value of the associated future is present.
                       This operation can also be used to enforce a particular scheduling on the execution graph.}];

    let arguments = (ins RAIRTensor:$value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}
//===----------------------------------------------------------------------===//
// NocOp
//===----------------------------------------------------------------------==
def NocOp : RAIR_Op<"noc", []> {
    let summary = "On-Chip Network Communication.";
    let description = [{TODO.}];

    let arguments = (ins RAIRTensor:$value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : RAIR_Op<"conv2d",
    [Pure]> {
    let summary = "RAIR dialect convolution 2d operation";
    let description = [{
        Performs a 2D convolution on the input tensor using specified kernel.
        Example:
        %output = "rair.conv2d"(%input, %kernel) {accelerator = "gemmini"}
          : (tensor<4x4xf32>, tensor<2x2xf32>) -> tensor<3x3xf32>
    }];

    let arguments = (ins RAIRTensor:$input, RAIRTensor:$kernel,
                         OptionalAttr<StrAttr>:$accelerator,
                         OptionalAttr<DenseI64ArrayAttr>:$tile_size,
                         OptionalAttr<StrAttr>:$dataflow);
    // let arguments = (ins RAIRTensor:$input, RAIRTensor:$kernel, RAIRTensor:$bias);

    let results = (outs RAIRTensor:$output);

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
def LaunchOp : RAIR_Op<"launch", [  RAIR_AsyncOpInterface, 
                                     RAIR_HierarchyInterface, 
                                     AttrSizedOperandSegments,  //MLIR变长操作数的分段管理                                  
                                     IsolatedFromAbove,//MLIR特性，标记操作的region是上层隔离的
                                     AffineScope,  //MLIR特性，为操作创建一个仿射作用域
                                     SingleBlockImplicitTerminator<"LaunchTerminatorOp">]>,
                          Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                         Variadic<RAIR_AsyncToken>:$async_dependencies,
                                         Variadic<Index>:$sizes,
                                         Variadic<AnyType>:$launch_operands,
                                         OptionalAttr<StrAttr>:$accelerator,
                                         OptionalAttr<FlatSymbolRefAttr>:$fallback)>,
                          Results<(outs Optional<RAIR_AsyncToken>:$async_token)>{
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
def LaunchTerminatorOp : RAIR_Op<"launch_terminator", [HasParent<"LaunchOp">,
                                                          Pure,
                                                          Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `rair.launch`.";
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
def HerdOp : RAIR_Op<"herd", [ RAIR_AsyncOpInterface, 
                                RAIR_HierarchyInterface, 
                                AttrSizedOperandSegments,                              
                                IsolatedFromAbove,
                                AffineScope,  
                                SingleBlockImplicitTerminator<"HerdTerminatorOp">]>,
                          Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                         OptionalAttr<StrAttr>:$link_with,
                                         Variadic<RAIR_AsyncToken>:$async_dependencies,
                                         Variadic<Index>:$sizes,
                                         Variadic<AnyType>:$herd_operands)>,                                        
                          Results<(outs Optional<RAIR_AsyncToken>:$async_token)>{
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
def HerdTerminatorOp : RAIR_Op<"herd_terminator", [HasParent<"HerdOp">,
                                                          Pure,
                                                          Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `rair.herd`.";
  let description = [{
    A terminator operation for the body of `rair.herd` operations.
    `rair.herd` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}
//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//
def KernelOp : RAIR_Op<"kernel", [
    SingleBlockImplicitTerminator<"KernelTerminatorOp">]> {
  let summary = "Structured accelerator kernel region";
  let description = [{
    Defines a structured kernel region for accelerator execution. The kernel
    encapsulates one or more compute operations that should be executed as a
    unit on the accelerator. Operations within this region are candidates for
    fusion, scheduling, and accelerator-side optimization.

    The `accelerator` attribute identifies which accelerator target this kernel
    is intended for. The optional `fallback` attribute names a CPU-path function
    to use if the kernel cannot be legally or profitably executed on the
    accelerator. The optional `tile_size` and `dataflow` attributes carry
    target-aware decomposition and scheduling preferences.

    Example:
    ```mlir
    rair.kernel attributes {accelerator = "gemmini",
                            fallback = @cpu_matmul,
                            tile_size = array<i64: 16, 16, 16>,
                            dataflow = "weight_stationary"} {
      rair.matmul ins(%a, %b : memref<128x128xf32>, memref<128x128xf32>)
                   outs(%c : memref<128x128xf32>)
      rair.kernel_terminator
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$accelerator,
    OptionalAttr<FlatSymbolRefAttr>:$fallback,
    OptionalAttr<DenseI64ArrayAttr>:$tile_size,
    OptionalAttr<StrAttr>:$dataflow
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{ attr-dict-with-keyword $body }];
}
//===----------------------------------------------------------------------===//
// KernelTerminatorOp
//===----------------------------------------------------------------------===//
def KernelTerminatorOp : RAIR_Op<"kernel_terminator", [HasParent<"KernelOp">,
                                                        Pure,
                                                        Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `rair.kernel`.";
  let description = [{
    A terminator operation for the body of `rair.kernel` operations.
    `rair.kernel` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}
//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//
def AllocOp: RAIR_Op<"alloc", []> {
  let summary = "RAIR Allocation Op";
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
def DeallocOp: RAIR_Op<"dealloc", []> {
  let summary = "RAIR Deallocation Op";
  let arguments = (ins AnyMemRef:$memref);

  let results = (outs);
  let assemblyFormat = [{
    attr-dict $memref `:` type($memref)
  }];
  let description = [{
    RAIR Deallocation Op
  }];
  // let hasCanonicalizer  = 1;
}
//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//
def TransferOp : RAIR_Op<"transfer", []> {
  let summary = "Transfer data between memory spaces";
  let description = [{
    Explicitly transfers data between memory spaces (e.g., host DRAM to
    accelerator scratchpad, or scratchpad to accumulator). Making transfers
    explicit in IR enables redundant transfer elimination, transfer/compute
    overlap, and memory lifetime analysis.

    Example:
    ```mlir
    rair.transfer %host_buf to %accel_buf : memref<128x128xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    OptionalAttr<StrAttr>:$src_memory_space,
    OptionalAttr<StrAttr>:$dst_memory_space
  );
  let results = (outs);
  let assemblyFormat = [{ $src `to` $dst attr-dict `:` type($src) `,` type($dst) }];
}
//===----------------------------------------------------------------------===//
// AllocBufferOp
//===----------------------------------------------------------------------===//
def AllocBufferOp : RAIR_Op<"alloc_buffer", []> {
  let summary = "Allocate accelerator-visible buffer";
  let description = [{
    Allocates a buffer in an accelerator-visible memory space (e.g., scratchpad,
    accumulator). The allocation is associated with an accelerator context,
    enabling buffer lifetime analysis and memory reuse across kernels.

    Example:
    ```mlir
    %buf = rair.alloc_buffer %ctx {memory_space = "scratchpad"}
           : memref<128x128xf32>
    ```
  }];

  let arguments = (ins
    RAIR_Context:$context,
    OptionalAttr<StrAttr>:$memory_space
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{ $context attr-dict `:` type($result) }];
}
//===----------------------------------------------------------------------===//
// DeallocBufferOp
//===----------------------------------------------------------------------===//
def DeallocBufferOp : RAIR_Op<"dealloc_buffer", []> {
  let summary = "Deallocate accelerator-visible buffer";
  let description = [{
    Deallocates a buffer previously allocated in accelerator-visible memory.
    Explicit deallocation enables deterministic buffer lifetime analysis.

    Example:
    ```mlir
    rair.dealloc_buffer %ctx, %buf : memref<128x128xf32>
    ```
  }];

  let arguments = (ins RAIR_Context:$context, AnyMemRef:$buffer);
  let results = (outs);
  let assemblyFormat = [{ $context `,` $buffer attr-dict `:` type($buffer) }];
}
//===----------------------------------------------------------------------===//
// Template of *_DRV  
//===----------------------------------------------------------------------===//
class RAIR_DRV_Op<string opName> : RAIR_Op<opName,
      [RAIR_AsyncOpInterface,
       RAIR_MemcpyInterface,
       AttrSizedOperandSegments]> {

  let summary = "dma operator";
  let arguments = (
    ins Variadic<RAIR_AsyncToken>:$async_dependencies,
        AnyRankedOrUnrankedMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyRankedOrUnrankedMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );

  let results = (outs Optional<RAIR_AsyncToken>:$async_token);

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
def LoadDrvOp : RAIR_DRV_Op<"load">;
def StoreDrvOp : RAIR_DRV_Op<"store">;


//===----------------------------------------------------------------------===//
// WaitAllOp
//===----------------------------------------------------------------------===//
def WaitAllOp: RAIR_Op<"wait_all", [RAIR_AsyncOpInterface]> {
  let arguments = (ins Variadic<RAIR_AsyncToken>:$async_dependencies);
  let results = (
    outs Optional<RAIR_AsyncToken>:$async_token
  );

  let summary = "wait for all operator";
  let description = [{
    Wait for all async tokens before preceding.
    Example:
        %0 = air.wait_all async 
        %4 = rair.wait_all async [%a4, %a5] 
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
// AcquireOp
//===----------------------------------------------------------------------===//
def AcquireOp : RAIR_Op<"acquire", []> {
  let summary = "Acquire accelerator execution context";
  let description = [{
    Acquires an accelerator execution context for a RoCC-style accelerator.
    The context represents resource ownership and must be released with
    `rair.release` when no longer needed.

    Example:
    ```mlir
    %ctx = rair.acquire {accelerator = "gemmini"} : !rair.context
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$accelerator);
  let results = (outs RAIR_Context:$context);
  let assemblyFormat = [{ attr-dict `:` type($context) }];
}
//===----------------------------------------------------------------------===//
// ReleaseOp
//===----------------------------------------------------------------------===//
def ReleaseOp : RAIR_Op<"release", []> {
  let summary = "Release accelerator execution context";
  let description = [{
    Releases a previously acquired accelerator context, freeing associated
    resources. All operations on the context must have completed before release.

    Example:
    ```mlir
    rair.release %ctx : !rair.context
    ```
  }];

  let arguments = (ins RAIR_Context:$context);
  let results = (outs);
  let assemblyFormat = [{ $context attr-dict `:` type($context) }];
}
//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//
def AwaitOp : RAIR_Op<"await", []> {
  let summary = "Wait for accelerator operations to complete";
  let description = [{
    Blocks until all pending accelerator operations associated with the given
    context have completed. This makes synchronization an explicit, analyzable
    program event rather than a hidden convention.

    Example:
    ```mlir
    rair.await %ctx : !rair.context
    ```
  }];

  let arguments = (ins RAIR_Context:$context);
  let results = (outs);
  let assemblyFormat = [{ $context attr-dict `:` type($context) }];
}
//===----------------------------------------------------------------------===//
// QuantizationOp
//===----------------------------------------------------------------------===//
// def QuantizationOp : RAIR_Op<"quantization",
//     [Pure]> {
//   let summary = "quantization operation";
//   let description = [{
//     y_quantized[i] = Round(a[i] - min) / step) * step + min
//         where, step = (max-min)/NoOfLevels 
    
//     The "quantization" operation steps:
//     1) calulate NoOfLevels = 2^NoOfBits -- separate
//     2) Then calculate stepSize = (Max-Min)/NoOfLevels
//     3) iterate for all the elements and calculate quantizedCoeff

//     GetLevelForVal =  (a[i] - min)/step
//     RoundedVal = arith.FPToSI(GetLevelForVal)
//     QuantVal = RoundedVal * step + min_val

//     ex: Original coefficients: [ 3.2 -1.5  0.8 -2.9  4.5] & numLevels = 16, max=5, min=-5
//         Quantized coefficients: [ 3.125 -1.25   0.625 -3.125  4.375]
//   }];

//   let arguments = (ins RAIRTensor:$input, RAIRTensor:$Nlevels, RAIRTensor:$max, RAIRTensor:$min);
//   let results = (outs RAIRTensor);

//   // Enable registering canonicalization patterns with this operation.
//   // let hasCanonicalizer = 1;

//   // Allow building a LowpassFilter with from the input operand.
//   let builders = [
//     OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$Nlevels, "mlir::Value":$max, "mlir::Value":$min)>
//   ];

//   // Indicate that additional verification for this operation is necessary.
//   // let hasVerifier = 1;
// }
// def DequantizationOp : RAIR_Op<"dequantization",
//     [Pure]> {
//   let summary = "dequantization operation";
//   let description = [{
//     a_recovered[i] = quant_data[i] * step + min
//         where, step = (max - min) / NoOfLevels 
    
//     The "dequantization" operation steps:
//     1) Use the same NoOfLevels as quantization process -- separate
//     2) Calculate stepSize = (Max - Min) / NoOfLevels
//     3) Iterate for all the elements and calculate recoveredCoeff

//     QuantLevel = quant_data[i]
//     RecoveredVal = QuantLevel * step + min_val

//     ex: Quantized coefficients: [ 3.125 -1.25   0.625 -3.125  4.375] & numLevels = 16, max=5, min=-5
//         Recovered coefficients: [ 3.125 -1.25   0.625 -3.125  4.375]
//   }];

//   let arguments = (ins RAIRTensor:$quant_input, RAIRTensor:$Nlevels, RAIRTensor:$max, RAIRTensor:$min);
//   let results = (outs RAIRTensor);

//   let builders = [
//     OpBuilder<(ins "mlir::Value":$quant_input, "mlir::Value":$Nlevels, "mlir::Value":$max, "mlir::Value":$min)>
//   ];

//   // let hasVerifier = 1;
// }

//===----------------------------------------------------------------------===//
// FuncOp 
//===----------------------------------------------------------------------===//

def FuncOp : RAIR_Op<"func", [
     FunctionOpInterface,IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "rair.func" operation represents a user defined function. These are
    callable SSA-region operations that contain RAIR computations.

    Example:

    ```mlir
    rair.func @main() {
      %0 = rair.constant dense<5.500000e+00> : tensor<f64>
      %1 = rair.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      rair.print %1 : tensor<2x2xf64>
      rair.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "::mlir::StringRef":$name, "::mlir::FunctionType":$type,
    CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}
//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : RAIR_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      rair.func @foo() -> tensor<2xf64> {
        ...
        rair.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<RAIRTensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std ::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// Base class for NCHW pooling ops   
//===----------------------------------------------------------------------===//
class PoolingNchwBaseOp<string mnemonic, string opDoc>
  : RAIR_Op<mnemonic, [Pure]> {

  let summary = opDoc;
  let description = [{
    The operation performs 2D pooling
    on an input tensor in NCHW layout.

    For each batch n and channel c:
      output[n, c, h, w] =
        reduction over kh, kw:
          input[n, c,
                h * stride_h + kh * dilation_h,
                w * stride_w + kw * dilation_w]

    All operands are memrefs. The result is written into `output`.

    Example:
    ```mlir
    rair.}] # mnemonic # [{
      {dilations = dense<[1, 1]> : vector<2xi64>,
       strides   = dense<[2, 2]> : vector<2xi64>}
      ins(%input, %kernel : memref<1x64x114x114xf32>, memref<3x3xf32>)
      outs(%output : memref<1x64x56x56xf32>)
    ```
  }];

  let arguments = (ins
    AnyMemRef:$input,
    AnyMemRef:$kernel,
    AnyMemRef:$output,
    TypedAttrInterface:$dilations,
    TypedAttrInterface:$strides,
    OptionalAttr<StrAttr>:$accelerator,
    OptionalAttr<DenseI64ArrayAttr>:$tile_size,
    OptionalAttr<StrAttr>:$dataflow
  );

  let results = (outs);

  let assemblyFormat = [{
    attr-dict
    `ins` `(` $input `,` $kernel `:`
      type($input) `,` type($kernel) `)`
    `outs` `(` $output `:` type($output) `)`
  }];
}
//===----------------------------------------------------------------------===//
// PoolingNchwMaxOp  对NCHW格式输入执行窗口内的元素取最大值的池化操作
//===----------------------------------------------------------------------===//
def PoolingNchwMaxOp
  : PoolingNchwBaseOp<"pooling_nchw_max",
                      "Performs NCHW max pooling.">;
//===----------------------------------------------------------------------===//
// PoolingNchwSumOp 对NCHW格式输入执行窗口内的元素求和的池化操作
//===----------------------------------------------------------------------===//
def PoolingNchwSumOp
  : PoolingNchwBaseOp<"pooling_nchw_sum",
                      "Performs NCHW sum pooling.">;
//===----------------------------------------------------------------------===//
// Conv2dNchwFchw   对每个输出位置，计算输入对应区域与卷积核的逐元素相乘再求和
//===----------------------------------------------------------------------===//
def Conv2dNchwFchwOp
  : PoolingNchwBaseOp<"conv_2d_nchw_fchw",
                      "Performs NCHW conv2d.">;
//===----------------------------------------------------------------------===//
// WorldOp 
//===----------------------------------------------------------------------===//
def WorldOp : RAIR_Op<"world", [Pure]> {
    let summary = "print RAIR, World";
    let description = [{
        The "world" operation prints "RAIR, World", and produces
        no results.
    }];
}

#endif // RISCV_OPS