"""
RoCC dialect for accelerator-side memref ops (batch GEMM, 2-D matmul, transpose).

MLIR operation names use the ``rocc.`` prefix. ``RISCV`` is kept as an alias of
``ROCC`` for older imports.
"""

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerType,
    MemRefType,
)
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
)


@irdl_op_definition
class BatchMatmulOp(IRDLOperation):
    """
    Batched matrix multiplication.

    Custom syntax: rocc.batch_matmul ins(%A, %B : memref<...>, memref<...>) outs(%C : memref<...>)

    Performs batched matrix multiplication: C[batch] = A[batch] @ B[batch]
    where A is (batch, M, K), B is (batch, K, N), C is (batch, M, N).
    """

    name = "rocc.batch_matmul"

    A = operand_def(MemRefType)
    B = operand_def(MemRefType)
    C = operand_def(MemRefType)
    result = result_def(MemRefType)

    m_dim = opt_prop_def(IntegerType)
    n_dim = opt_prop_def(IntegerType)
    k_dim = opt_prop_def(IntegerType)
    batch_dim = opt_prop_def(IntegerType)

    def __init__(
        self,
        A: SSAValue,
        B: SSAValue,
        C: SSAValue,
        m_dim: IntegerType | None = None,
        n_dim: IntegerType | None = None,
        k_dim: IntegerType | None = None,
        batch_dim: IntegerType | None = None,
    ):
        attrs = {}
        if m_dim is not None:
            attrs["m_dim"] = m_dim
        if n_dim is not None:
            attrs["n_dim"] = n_dim
        if k_dim is not None:
            attrs["k_dim"] = k_dim
        if batch_dim is not None:
            attrs["batch_dim"] = batch_dim

        super().__init__(
            operands=[A, B, C],
            result_types=[C.type],
            attributes=attrs,
        )


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """
    Two-operand matrix multiply (same operand layout as batch matmul, 2-D friendly).

    Custom syntax: rocc.matmul ins(%A, %B : memref<...>, memref<...>) outs(%C : memref<...>)
    """

    name = "rocc.matmul"

    A = operand_def(MemRefType)
    B = operand_def(MemRefType)
    C = operand_def(MemRefType)
    result = result_def(MemRefType)

    m_dim = opt_prop_def(IntegerType)
    n_dim = opt_prop_def(IntegerType)
    k_dim = opt_prop_def(IntegerType)
    batch_dim = opt_prop_def(IntegerType)

    def __init__(
        self,
        A: SSAValue,
        B: SSAValue,
        C: SSAValue,
        m_dim: IntegerType | None = None,
        n_dim: IntegerType | None = None,
        k_dim: IntegerType | None = None,
        batch_dim: IntegerType | None = None,
    ):
        attrs = {}
        if m_dim is not None:
            attrs["m_dim"] = m_dim
        if n_dim is not None:
            attrs["n_dim"] = n_dim
        if k_dim is not None:
            attrs["k_dim"] = k_dim
        if batch_dim is not None:
            attrs["batch_dim"] = batch_dim

        super().__init__(
            operands=[A, B, C],
            result_types=[C.type],
            attributes=attrs,
        )


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Tensor transpose with an explicit permutation attribute.

    Example:
      rocc.transpose ins(%input : memref<1x128x8x64xf32>) outs(%output : memref<1x8x128x64xf32>)
                   {permutation = array<i64: 0, 2, 1, 3>}
    """

    name = "rocc.transpose"

    input = operand_def(MemRefType)
    output = operand_def(MemRefType)
    permutation = opt_prop_def(ArrayAttr)

    def __init__(
        self, input: SSAValue, output: SSAValue, permutation: ArrayAttr | None = None
    ):
        super().__init__(
            operands=[input, output],
            result_types=[],
            attributes={"permutation": permutation} if permutation else {},
        )


ROCC = Dialect(
    "rocc",
    [
        BatchMatmulOp,
        MatmulOp,
        TransposeOp,
    ],
    [],
)

RISCV = ROCC
