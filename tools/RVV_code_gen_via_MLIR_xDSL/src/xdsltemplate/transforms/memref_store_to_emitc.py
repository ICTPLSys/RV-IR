from xdsl.context import Context
from xdsl.dialects import emitc, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCAssignOp, EmitCSubscriptOp


class ConvertMemrefStoreToEmitc(RewritePattern):
    """
    memref.store %value, %memref[%idx] : memref<?xf32>

    func.func @emitc_store(%A : !emitc.ptr<f32>, %idx : index, %val : f32) -> f32 {
            %subscript = emitc.subscript %A[%idx] : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>

            emitc.assign %val : f32 to %subscript : !emitc.lvalue<f32>
            # or
          # "emitc.assign"(%subscript, %val) : (!emitc.lvalue<f32>, f32) -> ()

          return %val : f32
    }

    """

    # @op_type_rewrite_pattern
    # def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter):
    #     base = op.memref
    #     value_to_store = op.value
    #
    #     indices: Sequence = list(op.indices)
    #     elem_ty: Attribute = op.results[0].type
    #     lvalue_ty = emitc.EmitC_LValueType(elem_ty)
    #
    #     subscript_op = EmitCSubscriptOp(base, indices[0], lvalue_ty)
    #     rewriter.insert_op_before_matched_op(subscript_op)
    #
    #
    #     assign_op = EmitCAssignOp(value_to_store, subscript_op.results[0])
    #     rewriter.replace_matched_op(assign_op)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter):
        base = op.memref
        indices = list(op.indices)
        value_to_store = op.value

        elem_ty = value_to_store.type
        lvalue_ty = emitc.EmitC_LValueType(elem_ty)

        # Get lvalue
        subscript_op = EmitCSubscriptOp(base, indices[0], lvalue_ty)
        rewriter.insert_op_before_matched_op(subscript_op)

        # Assign (void operation)
        assign_op = EmitCAssignOp(subscript_op.results[0], value_to_store)

        # Replace (both void)
        rewriter.replace_matched_op(assign_op)


class MemrefStoreToEmitcPass(ModulePass):
    name = "memref_store_to_emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            ConvertMemrefStoreToEmitc(),
            apply_recursively=True,
        )
        walker.rewrite_module(op)
