from xdsl.context import Context
from xdsl.dialects import emitc, memref

# from xdsl.xdsl.dialects.builtin import StringAttr
from xdsl.dialects.builtin import (
    Float32Type,
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

from xdsltemplate.dialects.emitc_ext import EmitCSubscriptOp
from xdsltemplate.dialects.rvv import *


def _emitc_ptr_operand_for_memref(val: SSAValue) -> SSAValue:
    """Use unrealized_conversion_cast from MemRefToEmitCPass (memref → !emitc.ptr)."""
    if isinstance(val.type, emitc.EmitC_PointerType):
        return val
    for use in val.uses:
        cast_op = use.operation
        if not isinstance(cast_op, UnrealizedConversionCastOp):
            continue
        if (
            cast_op.inputs
            and cast_op.inputs[0] is val
            and cast_op.results
            and isinstance(cast_op.results[0].type, emitc.EmitC_PointerType)
        ):
            return cast_op.results[0]
    return val


def _emitc_ptr_to_element(mem_val: SSAValue) -> emitc.EmitC_PointerType:
    """Type of &elem for one scalar in a memref-of-element or ptr-to-element."""
    t = mem_val.type
    if isinstance(t, memref.MemRefType):
        return emitc.EmitC_PointerType(t.element_type)
    if isinstance(t, emitc.EmitC_PointerType):
        return t
    return emitc.EmitC_PointerType(Float32Type())


class ConvertRVVSetvlToEmitC(RewritePattern):
    """
    Convert RVV to emitc
    such as Stevop
    %vl = "rvv.setvl"(%loop_size) : (index) -> index
    to
    %vl = emitc.call_opaque "__riscv_vsetvl_e32m1"(%loop_size) : (index) -> index
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SetvlOp, rewriter: PatternRewriter):
        # match and rewrite setvlop operation
        # we want to use call operation in emitc c to do the convertion
        call_op = emitc.EmitC_CallOpaqueOp(
            # There are error, check the function definition of the original back end code for Call op
            callee="__riscv_vsetvl_e32m1",  # str is fine; it will be wrapped as StringAttr
            call_args=[op.avl],  # avl defined in rvv dialetc, the input
            result_types=[op.vl.type],  # vl is the type
        )

        rewriter.replace_op(op, [call_op])

        # rewriter.replace_op(op, call_op.results)


class ConvertRVV_vle32_v_f32m1_ToEmitC(RewritePattern):
    """
    Lower
    We get
          %va = "rvv.vle32_v_f32m1Op"(%arg0, %vl) : (memref<?xf32>, index) -> f32
      secondery
          %va = "rvv.vle32_v_f32m1Op"(%arg0, %vl) : (!emitc.ptr<f32>, index) -> f32
        to
        %va = emitc.call_opaque "__riscv_vle32_v_f32m1"(%arg0, %vl)
              : (!emitc.ptr<f32>, index) -> !emitc.opaque<"vfloat32m1_t">
               Then need to be lowered emitc call op
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vle32_v_f32m1Op, rewriter: PatternRewriter):
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        base = _emitc_ptr_operand_for_memref(op.memref)
        subscript = EmitCSubscriptOp(base, op.offset, lvalue_type)

        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[_emitc_ptr_to_element(op.memref)],
            attributes={"applicableOperator": StringAttr("&")},
        )
        # // Custom form of applying the & operator.
        # %0 = emitc.apply "&"(%arg0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
        #
        # // Generic form of the same operation.
        # %0 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

        # Create RVV call with the address
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vle32_v_f32m1",
            call_args=[ptr_ref.result, op.avl],
            # result_types=[op.vl.type],
            result_types=[vector_type],
        )
        # Replace with all three operations
        rewriter.replace_op(op, [subscript, ptr_ref, call_op])


from xdsltemplate.dialects.emitc_ext import EmitCLoadOp


class ConvertRVV_vfmacc_vf_f32m1_ToEmitC(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vfmacc_vf_f32m1Op, rewriter: PatternRewriter):
        # Create subscript to get lvalue<f32>
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        base = _emitc_ptr_operand_for_memref(op.memref)
        subscript = EmitCSubscriptOp(base, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)

        # Load the lvalue to get f32 value
        load_op = EmitCLoadOp(src=subscript.result, result_type=Float32Type())
        rewriter.insert_op_before_matched_op(load_op)

        # Now use the loaded f32 value in the call
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vfmacc_vf_f32m1",
            call_args=[
                op.vd,
                load_op.result,
                op.vs,
                op.avl,
            ],  # Use load_op.result instead of subscript.result
            result_types=[vector_type],
        )

        rewriter.replace_op(op, call_op)


class ConvertRVV_vfmv_v_f_f32m1_ToEmitC(RewritePattern):
    """
        Convert:
        %v = "rvv.vfmv_v_f_f32m1"(%scalar, %vl) : (f32, index) -> !rvv.vec<f32, 1>
    To:
        %v = emitc.call_opaque "__riscv_vfmv_v_f_f32m1"(%scalar, %vl)
            : (f32, index) -> !emitc.opaque<"vfloat32m1_t">
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vfmv_v_f_f32m1Op, rewriter: PatternRewriter):
        vector_type = emitc.EmitC_OpaqueType(StringAttr("vfloat32m1_t"))
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vfmv_v_f_f32m1",
            call_args=[op.scalar, op.avl],
            result_types=[vector_type],
        )
        rewriter.replace_op(op, call_op)

        return


class ConvertRVV_vse32_v_f32m1_ToEmitC(RewritePattern):
    """
    Convert
        rvv.vse32_v_f32m1( %memeref, %offset, %vec, %vl) : (memref<?xf32>, index, !rvv.vfloat32m1, index) -> ()
    To
     __riscv_vse32_v_f32m1(&C[0], C_reg_0_0, (vl));

     To find the pointer via emitc-subscript function, get the lvalue of the memeref location acording to the offset
     Then use app function to return a emitc ptr
     then use emitc call function to call a c func with ptr address, vecter name and vl
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vse32_v_f32m1Op, rewriter: PatternRewriter):
        # Get address: &C[offset]
        lvalue_type = emitc.EmitC_LValueType(Float32Type())
        base = _emitc_ptr_operand_for_memref(op.memref)
        subscript = EmitCSubscriptOp(base, op.offset, lvalue_type)
        rewriter.insert_op_before_matched_op(subscript)

        ptr_ref = emitc.EmitC_ApplyOp(
            operands=[subscript.result],
            result_types=[_emitc_ptr_to_element(op.memref)],
            attributes={"applicableOperator": StringAttr("&")},
        )
        rewriter.insert_op_before_matched_op(ptr_ref)

        # Call the store intrinsic (no result)
        call_op = emitc.EmitC_CallOpaqueOp(
            callee="__riscv_vse32_v_f32m1",
            call_args=[ptr_ref.result, op.vec, op.avl],
            result_types=[],  # No result for store
        )

        rewriter.replace_op(op, call_op)


class RVVToEmitCPass(ModulePass):
    name = "rvv_to_emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            ConvertRVVSetvlToEmitC(),
            ConvertRVV_vle32_v_f32m1_ToEmitC(),
            ConvertRVV_vfmacc_vf_f32m1_ToEmitC(),
            ConvertRVV_vfmv_v_f_f32m1_ToEmitC(),
            ConvertRVV_vse32_v_f32m1_ToEmitC(),
        ]
        for pattern in patterns:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)
