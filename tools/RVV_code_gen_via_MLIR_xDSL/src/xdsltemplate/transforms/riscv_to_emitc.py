"""
Convert RAIR dialect operations to EmitC (batch GEMM, 2-D matmul, transpose).

``RISCVToEmitCPass`` is an alias of ``RAIRToEmitCPass`` for backward compatibility.
"""

from xdsl.context import Context
from xdsl.dialects import emitc
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCConstantOp
from xdsltemplate.dialects.riscv import BatchMatmulOp, MatmulOp, TransposeOp


def get_pointer_from_value(value: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """
    Get a pointer value from an SSA value.
    If the value is already a pointer type, return it.
    If it's a memref, look for or create an unrealized_conversion_cast to pointer.
    """
    if isinstance(value.type, emitc.EmitC_PointerType):
        return value

    if isinstance(value.type, MemRefType):
        for use in value.uses:
            if isinstance(use.operation, UnrealizedConversionCastOp):
                cast_result = use.operation.results[0]
                if isinstance(cast_result.type, emitc.EmitC_PointerType):
                    return cast_result

        raise ValueError(
            f"Memref value {value} does not have a pointer cast. "
            "Ensure memref-to-emitc pass runs before rair-to-emitc."
        )

    return value


def _rewrite_batch_like_gemm(op: BatchMatmulOp | MatmulOp, rewriter: PatternRewriter) -> None:
    A_type = op.A.type
    B_type = op.B.type

    if not isinstance(A_type, MemRefType):
        raise ValueError(
            f"Expected memref type for operand A, got {A_type}. "
            "Ensure rair.batch_matmul / rair.matmul is used before memref-to-emitc conversion."
        )

    A_shape = A_type.shape.data
    B_shape = B_type.shape.data

    batch = A_shape[0].data if len(A_shape) == 3 else 1
    M = A_shape[1].data if len(A_shape) == 3 else A_shape[0].data
    K = A_shape[2].data if len(A_shape) == 3 else A_shape[1].data
    N = B_shape[2].data if len(B_shape) == 3 else B_shape[1].data

    c_batch = EmitCConstantOp(IntegerAttr(batch, IndexType()))
    c_M = EmitCConstantOp(IntegerAttr(M, IndexType()))
    c_N = EmitCConstantOp(IntegerAttr(N, IndexType()))
    c_K = EmitCConstantOp(IntegerAttr(K, IndexType()))

    rewriter.insert_op_before_matched_op(c_batch)
    rewriter.insert_op_before_matched_op(c_M)
    rewriter.insert_op_before_matched_op(c_N)
    rewriter.insert_op_before_matched_op(c_K)

    ptr_A = get_pointer_from_value(op.A, rewriter)
    ptr_B = get_pointer_from_value(op.B, rewriter)
    ptr_C = get_pointer_from_value(op.C, rewriter)

    tensor_type = emitc.EmitC_OpaqueType(StringAttr("Tensor"))

    tensor_A = emitc.EmitC_CallOpaqueOp(
        callee="create_tensor_A",
        call_args=[ptr_A, c_M.results[0], c_K.results[0], c_batch.results[0]],
        result_types=[tensor_type],
    )
    rewriter.insert_op_before_matched_op(tensor_A)

    tensor_B = emitc.EmitC_CallOpaqueOp(
        callee="create_tensor_B",
        call_args=[ptr_B, c_K.results[0], c_N.results[0], c_batch.results[0]],
        result_types=[tensor_type],
    )
    rewriter.insert_op_before_matched_op(tensor_B)

    tensor_C = emitc.EmitC_CallOpaqueOp(
        callee="create_tensor_C",
        call_args=[ptr_C, c_M.results[0], c_N.results[0], c_batch.results[0]],
        result_types=[tensor_type],
    )
    rewriter.insert_op_before_matched_op(tensor_C)

    c0 = EmitCConstantOp(IntegerAttr(0, IndexType()))
    rewriter.insert_op_before_matched_op(c0)

    gemm_call = emitc.EmitC_CallOpaqueOp(
        callee="gemm_operator",
        call_args=[
            tensor_A.results[0],
            tensor_B.results[0],
            tensor_C.results[0],
            tensor_C.results[0],
            c0.results[0],
            c0.results[0],
        ],
        result_types=[],
    )

    if len(op.results) == 1:
        result_value = get_pointer_from_value(op.C, rewriter)
        rewriter.replace_op(op, [gemm_call], [result_value])
    elif len(op.results) == 0:
        rewriter.replace_op(op, [gemm_call], [])
    else:
        raise ValueError(
            f"Unexpected number of results on batch-like GEMM op: {len(op.results)}"
        )


class ConvertBatchMatmulToEmitC(RewritePattern):
    """Convert ``rair.batch_matmul`` to ``emitc.call_opaque`` calling ``gemm_operator``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BatchMatmulOp, rewriter: PatternRewriter):
        _rewrite_batch_like_gemm(op, rewriter)


class ConvertMatmulToEmitC(RewritePattern):
    """Convert ``rair.matmul`` to the same ``gemm_operator`` lowering as batch matmul."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter):
        _rewrite_batch_like_gemm(op, rewriter)


class ConvertTransposeToEmitC(RewritePattern):
    """Convert ``rair.transpose`` to ``transpose_operator``."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        input_type = op.input.type
        output_type = op.output.type

        if not isinstance(input_type, MemRefType):
            raise ValueError(
                f"Expected memref type for transpose operand input, got {input_type}. "
                "Ensure rair.transpose is used before memref-to-emitc conversion."
            )

        input_shape = input_type.shape.data
        output_shape = output_type.shape.data

        input_dim_constants = []
        for dim in input_shape:
            dim_value = dim.data
            c_dim = EmitCConstantOp(IntegerAttr(dim_value, IndexType()))
            rewriter.insert_op_before_matched_op(c_dim)
            input_dim_constants.append(c_dim.results[0])

        output_dim_constants = []
        for dim in output_shape:
            dim_value = dim.data
            c_dim = EmitCConstantOp(IntegerAttr(dim_value, IndexType()))
            rewriter.insert_op_before_matched_op(c_dim)
            output_dim_constants.append(c_dim.results[0])

        ptr_input = get_pointer_from_value(op.input, rewriter)
        ptr_output = get_pointer_from_value(op.output, rewriter)

        tensor_type = emitc.EmitC_OpaqueType(StringAttr("Tensor"))

        tensor_in = emitc.EmitC_CallOpaqueOp(
            callee="create_tensor_transpose_in",
            call_args=[ptr_input] + input_dim_constants,
            result_types=[tensor_type],
        )
        rewriter.insert_op_before_matched_op(tensor_in)

        tensor_out = emitc.EmitC_CallOpaqueOp(
            callee="create_tensor_transpose_out",
            call_args=[ptr_output] + output_dim_constants,
            result_types=[tensor_type],
        )
        rewriter.insert_op_before_matched_op(tensor_out)

        if op.permutation is not None:
            try:
                perm_values = []
                if hasattr(op.permutation, "data"):
                    for attr in op.permutation.data:
                        if hasattr(attr, "data"):
                            perm_values.append(int(attr.data))
                        else:
                            perm_values.append(int(attr))
                else:
                    perm_values = [1, 0]

                dim_axis = perm_values[1] if len(perm_values) > 1 else 1
            except (IndexError, ValueError, TypeError):
                dim_axis = 1

            c_dim_axis = EmitCConstantOp(IntegerAttr(dim_axis, IntegerType(64)))
            rewriter.insert_op_before_matched_op(c_dim_axis)
        else:
            c_dim_axis = EmitCConstantOp(IntegerAttr(1, IntegerType(64)))
            rewriter.insert_op_before_matched_op(c_dim_axis)

        transpose_call = emitc.EmitC_CallOpaqueOp(
            callee="transpose_operator",
            call_args=[
                tensor_in.results[0],
                tensor_out.results[0],
                c_dim_axis.results[0],
            ],
            result_types=[],
        )

        rewriter.replace_op(op, transpose_call)


class RAIRToEmitCPass(ModulePass):
    """Convert all RAIR dialect operations to EmitC."""

    name = "rair-to-emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            ConvertBatchMatmulToEmitC(),
            ConvertMatmulToEmitC(),
            ConvertTransposeToEmitC(),
        ]
        for pattern in patterns:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)


RISCVToEmitCPass = RAIRToEmitCPass
