from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.gemm import AcolOp, BcolOp, CcolOp


class ConvertGemmAcolToArith(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: AcolOp | BcolOp | CcolOp, rewriter: PatternRewriter
    ):
        i = op.i
        j = op.j
        ld = op.operands[2]  # third operands, these can be three of them

        mul_op = arith.MuliOp(j, ld)
        add_op = arith.AddiOp(i, mul_op.result)

        rewriter.insert_op_before_matched_op([mul_op, add_op])
        rewriter.replace_matched_op([], [add_op.result])


class GemmToArithPass(ModulePass):
    name = "gemm-to-arith"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            ConvertGemmAcolToArith(),
        ]
        for pattern in patterns:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)
