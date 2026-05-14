from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import emitc, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCLoadOp, EmitCSubscriptOp


class ConvertMemrefLoadToEmitc(RewritePattern):
    """Lower memref.load → emitc.subscript + emitc.load
    such as
        %v0 = memref.load %arg0[%c0] : memref<?xf32>
    lower to
        %lv0 = "emitc.subscript"(%arg0, %c0) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        %v0 = "emitc.load"(%lv0) : (!emitc.lvalue<f32>) -> f32
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter):
        # define base operation or type during checking process
        base = op.memref
        indices: Sequence = list(op.indices)
        # obtain the data type
        elem_ty: Attribute = op.results[0].type

        # checking is the type is already emitc pointer type, is yes, do nothing
        if not isinstance(base.type, emitc.EmitC_PointerType):
            return  # do nothing

        # if its memref type, keep on

        # since we need to create a lv type used in subscript
        # we want to create a type refering to !emitc.lvalue<f32>
        lvalue_ty = emitc.EmitC_LValueType(elem_ty)

        # to implement
        # %lv0 = "emitc.subscript"(%arg0, %c0) : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
        subscript_op = EmitCSubscriptOp(base, indices[0], lvalue_ty)

        # %v0 = "emitc.load"(%lv0) : (!emitc.lvalue<f32>) -> f32
        # In this step, lvalue need to be converted to value in load operations
        load_op = EmitCLoadOp(subscript_op.results[0], elem_ty)

        # formal rewriter function
        # replace memref.load with subscript
        rewriter.replace_matched_op(
            [subscript_op, load_op], new_results=[load_op.results[0]]
        )


# pass registeration, need to apply convertion pattrn
class MemrefLoadToEmitcPass(ModulePass):
    name = "memref_load_to_emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            ConvertMemrefLoadToEmitc(),
            apply_recursively=True,
        )
        walker.rewrite_module(op)
