# xdsltemplate/transforms/scf_to_emitc.py
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, scf
from xdsl.dialects.builtin import Float32Type, FloatAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCLoadOp
from xdsltemplate.dialects.rvv import RVVFloat32M1Type


@irdl_op_definition
class EmitCVariableOp(IRDLOperation):
    name = "emitc.variable"
    res = result_def(emitc.EmitC_LValueType)


@irdl_op_definition
class EmitCAssignOp(IRDLOperation):
    name = "emitc.assign"


@irdl_op_definition
class EmitCForOp(IRDLOperation):
    name = "emitc.for"


@irdl_op_definition
class EmitCYieldOp(IRDLOperation):
    name = "emitc.yield"


def convert_type_for_emitc_variable(ty):
    """Convert RVV types to emitc.opaque, others stay as-is."""
    if isinstance(ty, RVVFloat32M1Type):
        return emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
    if hasattr(ty, "name") and "rvv" in str(ty.name).lower():
        return emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
    return ty


def get_emitc_variable_value_attr(ty):
    """Get appropriate value attribute for emitc.variable."""
    if isinstance(ty, emitc.EmitC_OpaqueType):
        return emitc.EmitC_OpaqueAttr(StringAttr(""))
    return FloatAttr(0.0, Float32Type())


class ConvertScfForToEmitCFor(RewritePattern):
    """
    Convert scf.for with iter_args into emitc.for + emitc.variable/assign structure.
    Supports multiple iter_args.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter):
        # print(f"[SCF] Pattern matched! iter_args count: {len(op.iter_args)}")

        if len(op.iter_args) == 0:
            # Simple for loop without iter_args - just convert directly
            return

        # === Step 1: Create variables for each iter_arg ===
        emitc_vars = []  # List of (var_op, inner_type)

        for i, init_acc in enumerate(op.iter_args):
            acc_ty = init_acc.type
            emitc_inner_ty = convert_type_for_emitc_variable(acc_ty)
            emitc_lvalue_ty = emitc.EmitC_LValueType(emitc_inner_ty)
            value_attr = get_emitc_variable_value_attr(emitc_inner_ty)

            # Create variable
            emitc_var = EmitCVariableOp.create(
                operands=[],
                result_types=[emitc_lvalue_ty],
                attributes={"value": value_attr},
            )
            rewriter.insert_op_before_matched_op(emitc_var)

            # Initialize with initial value
            init_assign = EmitCAssignOp.create(
                operands=[emitc_var.results[0], init_acc],
                result_types=[],
            )
            rewriter.insert_op_before_matched_op(init_assign)

            emitc_vars.append((emitc_var, emitc_inner_ty))
            # print(f"[SCF] Created variable {i} with type {emitc_lvalue_ty}")

        # === Step 2: Build new block for emitc.for ===
        lb, ub, step = op.lb, op.ub, op.step
        old_block = op.body.blocks[0]

        # New block only has induction variable (no iter_args as block args)
        new_block = Block(arg_types=[old_block.args[0].type])
        if old_block.args[0].name_hint:
            new_block.args[0].name_hint = old_block.args[0].name_hint

        # Value mapping: induction variable
        value_mapping: dict[SSAValue, SSAValue] = {
            old_block.args[0]: new_block.args[0]
        }

        # Get iter_arg block arguments (args after induction variable)
        iter_arg_block_args = list(old_block.args[1:])  # Skip induction var
        acc_loaded = [False] * len(iter_arg_block_args)

        # === Step 3: Clone operations with proper value mapping ===
        for inner_op in list(old_block.ops):
            # Skip the original scf.yield - we'll handle it separately
            if isinstance(inner_op, scf.YieldOp):
                # Create assign ops for each yielded value
                for i, yielded_val in enumerate(inner_op.operands):
                    # Map the yielded value through our value_mapping
                    mapped_val = value_mapping.get(yielded_val, yielded_val)
                    emitc_assign = EmitCAssignOp.create(
                        operands=[emitc_vars[i][0].results[0], mapped_val],
                        result_types=[],
                    )
                    new_block.add_op(emitc_assign)
                continue

            # Check if this op uses any iter_arg that hasn't been loaded yet
            for i, iter_arg in enumerate(iter_arg_block_args):
                if not acc_loaded[i]:
                    uses_this_arg = any(
                        operand == iter_arg for operand in inner_op.operands
                    )
                    if uses_this_arg:
                        # Load this accumulator
                        emitc_var, emitc_inner_ty = emitc_vars[i]
                        load_op = EmitCLoadOp(
                            src=emitc_var.results[0], result_type=emitc_inner_ty
                        )
                        new_block.add_op(load_op)
                        value_mapping[iter_arg] = load_op.results[0]
                        acc_loaded[i] = True
                        # print(f"[SCF] Loaded accumulator {i}")

            # Clone with mapping
            new_op = inner_op.clone(value_mapping)
            new_block.add_op(new_op)

            # Update mapping for results
            for old_res, new_res in zip(inner_op.results, new_op.results):
                value_mapping[old_res] = new_res

        # Add emitc.yield at end
        emitc_yield = EmitCYieldOp.create(operands=[], result_types=[])
        new_block.add_op(emitc_yield)

        # === Step 4: Create emitc.for ===
        new_region = Region()
        new_region.add_block(new_block)

        emitc_for = EmitCForOp.create(
            operands=[lb, ub, step],
            result_types=[],
            regions=[new_region],
        )

        # === Step 5: Create final loads for all accumulators ===
        final_loads = []
        for emitc_var, emitc_inner_ty in emitc_vars:
            final_load = EmitCLoadOp(
                src=emitc_var.results[0], result_type=emitc_inner_ty
            )
            final_loads.append(final_load)

        # === Step 6: Replace original op ===
        # print(f"[SCF] Replacing scf.for with emitc.for + {len(final_loads)} loads")
        rewriter.replace_op(
            op,
            [emitc_for] + final_loads,
            new_results=[fl.results[0] for fl in final_loads],
        )


class SCFToEmitCPass(ModulePass):
    name = "convert-scf-to-emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertScfForToEmitCFor(),
            apply_recursively=True,
        ).rewrite_module(op)