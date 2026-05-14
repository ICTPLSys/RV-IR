# src/xdsltemplate/transforms/memref_to_emitc.py

from xdsl.context import Context
from xdsl.dialects import emitc, func, memref
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class ConvertMemRefBlockArgsToPtr(RewritePattern):
    """Convert memref block arguments to emitc.ptr using unrealized_conversion_cast"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        # Check if any argument is a memref
        has_memref = any(
            isinstance(arg.type, memref.MemRefType) for arg in op.body.block.args
        )

        if not has_memref:
            return

        # For each memref argument, create a cast operation at the beginning of the block
        for arg in op.body.block.args:
            if isinstance(arg.type, memref.MemRefType):
                # Check if there's already a cast
                already_has_cast = False
                for use in arg.uses:
                    if isinstance(use.operation, UnrealizedConversionCastOp):
                        already_has_cast = True
                        break

                if already_has_cast:
                    continue

                # Create pointer type
                elem_type = arg.type.element_type
                ptr_type = emitc.EmitC_PointerType(elem_type)

                # Create cast operation
                cast_op = UnrealizedConversionCastOp(
                    operands=[arg], result_types=[ptr_type]
                )

                # Insert at the beginning of the block
                # Get first operation if exists
                first_op = None
                for block_op in op.body.block.ops:
                    first_op = block_op
                    break

                if first_op is not None:
                    rewriter.insert_op([cast_op], InsertPoint.before(first_op))
                else:
                    # Insert at end of block (InsertPoint.end(block))
                    rewriter.insert_op(
                        [cast_op], InsertPoint.at_block_end(op.body.block)
                    )


class ConvertMemRefAllocToPtr(RewritePattern):
    """Convert memref.alloc results to emitc.ptr using unrealized_conversion_cast

    This creates a cast that can be used by emitc operations that need pointers.
    The cast is not automatically applied to all uses - instead, operations that
    need pointers can explicitly use the cast result.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter):
        # Check if result is memref type
        result_type = op.results[0].type
        if not isinstance(result_type, memref.MemRefType):
            return

        # Check if we already created a cast for this alloc
        # (to avoid infinite loops)
        for use in op.results[0].uses:
            if isinstance(use.operation, UnrealizedConversionCastOp):
                # Already converted, skip
                return

        # Create pointer type
        elem_type = result_type.element_type
        ptr_type = emitc.EmitC_PointerType(elem_type)

        # Insert an unrealized_conversion_cast to convert memref to pointer
        cast_op = UnrealizedConversionCastOp(
            operands=[op.results[0]], result_types=[ptr_type]
        )

        # Insert the cast after the alloc op
        rewriter.insert_op_after_matched_op(cast_op)


class ConvertMemRefTypeToEmitCPtr(RewritePattern):
    """Convert memref types to emitc.ptr in function signatures"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        # Check if any argument is a memref
        has_memref = any(
            isinstance(arg.type, memref.MemRefType) for arg in op.body.block.args
        )

        if not has_memref:
            return

        # Convert argument types
        new_arg_types = []
        for arg in op.body.block.args:
            if isinstance(arg.type, memref.MemRefType):
                elem_type = arg.type.element_type
                # EmitC_PointerType takes a single type parameter
                new_type = emitc.EmitC_PointerType(elem_type)
                new_arg_types.append(new_type)
            else:
                new_arg_types.append(arg.type)

        # Convert output types
        new_output_types = []
        for output in op.function_type.outputs.data:
            if isinstance(output, memref.MemRefType):
                elem_type = output.element_type
                new_type = emitc.EmitC_PointerType(elem_type)
                new_output_types.append(new_type)
            else:
                new_output_types.append(output)

        # Create new function type
        new_func_type = func.FunctionType.from_lists(
            new_arg_types,
            new_output_types,
        )

        # Create new block with updated argument types
        new_block = Block(arg_types=new_arg_types)

        # Map old block arguments to new ones
        for old_arg, new_arg in zip(op.body.block.args, new_block.args):
            old_arg.replace_by(new_arg)

        # Move operations from old block to new block
        for op_inner in list(op.body.block.ops):
            op_inner.detach()
            new_block.add_op(op_inner)

        # Create new region with the new block
        new_region = Region()
        new_region.add_block(new_block)

        # Create new function operation
        new_func = func.FuncOp(
            name=op.sym_name.data,
            function_type=new_func_type,
            region=new_region,
        )

        # Copy other attributes
        for attr_name, attr_value in op.attributes.items():
            if attr_name not in ("sym_name", "function_type"):
                new_func.attributes[attr_name] = attr_value

        # Replace the old function with the new one
        # func.FuncOp has no results, so we pass empty list
        rewriter.replace_op(op, new_func, new_results=[])


class RemoveUnrealizedConversionCasts(RewritePattern):
    """Remove unrealized_conversion_cast operations that are no-ops or can be simplified"""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnrealizedConversionCastOp, rewriter: PatternRewriter
    ):
        # If the cast is from emitc.ptr to emitc.ptr (no-op), replace all uses with the input
        if isinstance(op.inputs[0].type, emitc.EmitC_PointerType) and isinstance(
            op.results[0].type, emitc.EmitC_PointerType
        ):
            # This is a no-op cast, replace all uses with the input
            op.results[0].replace_by(op.inputs[0])
            rewriter.erase_matched_op()
        # If the cast is from memref to emitc.ptr, we need to handle it differently
        # For now, keep it - we'll handle it in post-processing
        elif isinstance(op.inputs[0].type, memref.MemRefType) and isinstance(
            op.results[0].type, emitc.EmitC_PointerType
        ):
            # Keep this cast - it will be handled in C code generation
            return


class ConvertMemRefFuncSignatures(ModulePass):
    """Pass to remove unrealized conversion casts after function signature conversion"""

    name = "remove-unrealized-casts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            RemoveUnrealizedConversionCasts(), apply_recursively=True
        ).rewrite_module(op)


class MemRefToEmitCPass(ModulePass):
    """Lower memref in function signatures to !emitc.ptr, then insert casts for remaining memrefs."""

    name = "memref-to-emitc-casts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # Must run first: mlir-to-cpp cannot emit memref in func signatures.
        PatternRewriteWalker(
            ConvertMemRefTypeToEmitCPtr(), apply_recursively=True
        ).rewrite_module(op)

        # memref block args (e.g. inner funcs) → unrealized cast to ptr if any remain
        PatternRewriteWalker(
            ConvertMemRefBlockArgsToPtr(), apply_recursively=True
        ).rewrite_module(op)

        # Convert memref.alloc to emitc.ptr
        PatternRewriteWalker(
            ConvertMemRefAllocToPtr(), apply_recursively=True
        ).rewrite_module(op)
