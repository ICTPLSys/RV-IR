from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import (
    EmitCAddOp,
    EmitCConstantOp,
    EmitCMullOp,
    EmitCSubOp,
)


class ConvertArithConstantToEmitC(RewritePattern):
    """Rewrite `arith.constant` to `emitc.constant`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter):
        value_attr = op.value
        value_type = op.results[0].type

        # some fp32 constance i fail to convert, add here
        if isinstance(value_type, (builtin.Float32Type, builtin.Float64Type)):
            new_op = EmitCConstantOp(value=value_attr)
            rewriter.replace_matched_op(new_op)
            return

        # # Determine EmitC constant type (for index, use emitc.size_t)
        # if isinstance(value_type, builtin.IndexType):
        #     emitc_type = emitc.IndexType()
        # elif isinstance(value_type, builtin.IntegerType):
        #     emitc_type = emitc.IntegerType()
        # else:
        #     return  # skip unsupported types

        new_op = EmitCConstantOp(value=value_attr)
        rewriter.replace_matched_op(new_op)


class ConvertArithMulfOpToEmitC(RewritePattern):
    "Rewrite arith.mulf to emitc.mul and emit.add"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MulfOp, rewriter: PatternRewriter):
        lhs = op.lhs
        rhs = op.rhs
        # result_type = op.results[0].type

        new_op = EmitCMullOp(lhs, rhs)
        rewriter.replace_matched_op(new_op)


class ConvertArithMuliOpToEmitC(RewritePattern):
    "Rewrite arith.muli to emitc.mul and emit.add"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MuliOp, rewriter: PatternRewriter):
        lhs = op.lhs
        rhs = op.rhs
        new_op = EmitCMullOp(lhs, rhs)
        rewriter.replace_matched_op(new_op)


class ConvertArithAddOpToEmitC(RewritePattern):
    "rewrite arith.addf to emitc.add"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddfOp, rewriter: PatternRewriter):
        lhs = op.lhs
        rhs = op.rhs
        # result_type = op.results[0].type

        new_op = EmitCAddOp(lhs, rhs)
        rewriter.replace_matched_op(new_op)


class ConvertArithAddiOpToEmitC(RewritePattern):
    "rewrite arith.addi to emitc.add"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter):
        lhs = op.lhs
        rhs = op.rhs
        # result_type = op.results[0].type

        new_op = EmitCAddOp(lhs, rhs)
        rewriter.replace_matched_op(new_op)


class ConvertArithSubiOpToEmitC(RewritePattern):
    "rewrite arith.subi to emitc.sub"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SubiOp, rewriter: PatternRewriter):
        lhs = op.lhs
        rhs = op.rhs

        new_op = EmitCSubOp(lhs, rhs)
        rewriter.replace_matched_op(new_op)


class ConvertArithIndexCastToEmitC(RewritePattern):
    """Erase arith.index_cast by replacing result with input.

    In C output, int <-> size_t conversion is implicit, so no cast op needed.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.IndexCastOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op([], [op.input])


# class ConvertArithAddOpToEmitC(RewritePattern):
#     "rewrite arith.addf to emitc.add"
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, op: arith.AddfOp, rewriter: PatternRewriter):
#         lhs = op.lhs
#         rhs = op.rhs
#         # result_type = op.results[0].type
#
#         new_op = EmitCAddOp(lhs, rhs)
#         rewriter.replace_matched_op(new_op)


class ArithToEmitCPass(ModulePass):
    """Convert simple arith operations to EmitC dialect."""

    name = "arith-to-emitc"

    def apply(self, ctx: Context, module: ModuleOp) -> None:
        # rewriter = ConvertArithConstantToEmitC()
        patterns = [
            ConvertArithConstantToEmitC(),
            ConvertArithMulfOpToEmitC(),
            ConvertArithMuliOpToEmitC(),
            ConvertArithAddOpToEmitC(),
            ConvertArithAddiOpToEmitC(),
            ConvertArithSubiOpToEmitC(),
            ConvertArithIndexCastToEmitC(),
        ]

        from xdsl.pattern_rewriter import PatternRewriteWalker

        for pattern in patterns:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(module)
