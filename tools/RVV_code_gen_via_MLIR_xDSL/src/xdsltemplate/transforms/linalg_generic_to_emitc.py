"""
Convert Linalg Generic operations to EmitC
This pass converts linalg.generic operations to emitc.call_opaque calling NPU SDK operators
"""

from xdsl.context import Context
from xdsl.dialects import arith, emitc, linalg, math, memref
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    StringAttr,
)

# Import the correct emitc operation
from xdsl.dialects.emitc import EmitC_CallOpaqueOp
from xdsl.ir import BlockArgument, Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsltemplate.dialects.emitc_ext import EmitCConstantOp


def _memref_axis_to_npu_sdk_reduce_dim(memref_axis: int, rank: int) -> int:
    """
    Map an MLIR memref dimension index (outermost = 0) to the NPU ``Tensor``
    axis index used in ``reduce_dimN_*`` / ``make_tensor`` lowering.

    Must stay consistent with ``linalg_mlir_to_c._npu_view_dims``: the last
    up-to-three memref extents are packed as
    ``Tensor.dim0 = innermost``, ``Tensor.dim2 = outermost`` of that triple.
    """
    if rank <= 0:
        return 0
    # Slot within the 3-element tail after leading 1-padding (see _npu_view_dims).
    tail_slot = memref_axis - (rank - 3)
    if 0 <= tail_slot <= 2:
        return 2 - tail_slot
    # Axis outside the trailing-3 view; keep legacy clamp.
    return max(0, min(2, memref_axis))


class ConvertLinalgGenericToSub(RewritePattern):
    """Convert linalg.generic with arith.subf to emitc.call_opaque calling tensor_tensor_operator with OPERATION_SUB"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a simple element-wise operation
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return  # Not an element-wise operation

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 2 or len(outputs) != 1:
            return  # Not a binary operation

        # Check body contains arith.subf + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check if first operation is subf
        sub_op = body_ops[0]
        if "SUBF" not in type(sub_op).__name__.upper():
            return  # Not a subtraction operation

        # Get operands
        A = inputs[0]
        B = inputs[1]
        C = outputs[0]

        # Create constant for OPERATION_SUB (value should match enum definition)
        # TODO: Adjust this value based on the actual enum definition in the header file
        c_op_type = EmitCConstantOp(IntegerAttr(1, IntegerType(32)))
        rewriter.insert_op_before_matched_op(c_op_type)

        # Create a call to tensor_tensor_operator with OPERATION_SUB
        # Returns int status code (but we ignore it for now)
        call_op = EmitC_CallOpaqueOp(
            "tensor_tensor_operator",
            [A, B, C, c_op_type.results[0]],
            [IntegerType(32)],  # Returns int status
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("uint32_t"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToAdd(RewritePattern):
    """
    Convert linalg.generic with arith.addf to emitc.call_opaque calling tensor_tensor_operator with OPERATION_ADD

    Handles two cases:
    1. Tensor + Tensor (two inputs) -> tensor_tensor_operator(&in1, &in2, &out, OPERATION_ADD)
    2. Tensor + Constant (tensor + scalar) -> create constant tensor, then tensor_tensor_operator
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a simple element-wise add pattern
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return  # Not an element-wise operation

        inputs = op.inputs
        outputs = op.outputs

        # Check body contains arith.addf + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check if first operation is addf
        add_op = body_ops[0]
        is_add = isinstance(add_op, arith.AddfOp)

        # If not arith.AddfOp, check if it's emitc.add (already converted)
        if not is_add:
            op_type_name = type(add_op).__name__
            is_add = (
                "ADD" in op_type_name.upper() or "emitc.add" in str(add_op.name).lower()
            )

        if not is_add:
            return  # Not an add operation

        # Handle different cases based on number of inputs
        if len(inputs) == 2 and len(outputs) == 1:
            # Case 1: Tensor + Tensor
            A = inputs[0]
            B = inputs[1]
            C = outputs[0]

            # Create constant for OPERATION_ADD (value should match enum definition)
            # TODO: Adjust this value based on the actual enum definition in the header file
            c_op_type = EmitCConstantOp(IntegerAttr(0, IntegerType(32)))
            rewriter.insert_op_before_matched_op(c_op_type)

            # Create a call to tensor_tensor_operator with OPERATION_ADD
            # Returns int status code (but we ignore it for now)
            call_op = EmitC_CallOpaqueOp(
                "tensor_tensor_operator",
                [A, B, C, c_op_type.results[0]],
                [IntegerType(32)],  # Returns int status
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                        StringAttr("uint32_t"),
                    ]
                ),
            )
            rewriter.insert_op_before_matched_op(call_op)
            rewriter.erase_matched_op()
        elif len(inputs) >= 2 and len(outputs) == 1:
            # Case 2: Tensor + Constant (or more complex case)
            # This handles the epsilon add case: tensor + scalar
            # For now, we'll skip this complex case and let it fall through
            # The scalar add will be handled differently
            return
        else:
            return


class ConvertLinalgGenericToIdentity(RewritePattern):
    """
    Convert linalg.generic with linalg.yield %in (identity/copy) to emitc.call_opaque

    This handles:
    - Copy into ``memref.subview`` of a larger buffer: ``subview_rowwise_copy_add``
      (src, subview, parent) for C codegen that uses row-wise ``tensor_tensor_operator`` with
      ``OPERATION_ADD`` so compact src strides match the strided destination (MLIR offset/strides).
    - Simple dense copy: ``copy_operator`` (tensor_in -> tensor_out)
    - Broadcast: expanding dimensions (e.g., 2D -> 3D) -> ``broadcast_operator``
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is an element-wise operation
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return  # Not an element-wise operation

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return  # Not a single input/output operation

        # Check body contains only linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) != 1:
            return  # Should only have yield

        # Check if it's linalg.yield
        yield_op = body_ops[0]
        if "YIELD" not in type(yield_op).__name__.upper():
            return  # Not a yield operation

        # Check if yield returns the input directly
        yield_operands = yield_op.operands
        if len(yield_operands) != 1:
            return

        # The yielded value should be the block argument (input)
        yield_val = yield_operands[0]
        if not isinstance(yield_val, BlockArgument):
            return  # Not directly yielding the input

        A = inputs[0]
        C = outputs[0]

        # Check if shapes are different (broadcast/reshape)
        input_type = inputs[0].type
        output_type = outputs[0].type

        if not isinstance(input_type, MemRefType) or not isinstance(
            output_type, MemRefType
        ):
            return

        input_shape = input_type.shape.data
        output_shape = output_type.shape.data

        # Copy into a memref.subview of a larger parent (e.g. 4x4 tile into 10x10 pad).
        # Must not lower to a single tensor_tensor_add: compact src vs strided dst strides differ.
        is_subview_dest = isinstance(C, OpResult) and isinstance(
            C.owner, memref.SubviewOp
        )

        # Determine if this is a copy or broadcast operation
        if input_shape == output_shape and is_subview_dest:
            parent = C.owner.source
            op_name = "subview_rowwise_copy_add"
            call_op = EmitC_CallOpaqueOp(
                op_name,
                [A, C, parent],
                [C.type],
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                    ]
                ),
            )
        elif input_shape == output_shape:
            # Simple copy (same dense layout for src and dst)
            op_name = "copy_operator"
            call_op = EmitC_CallOpaqueOp(
                op_name,
                [A, C],
                [C.type],
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                    ]
                ),
            )
        else:
            # Broadcast/reshape
            op_name = "broadcast_operator"
            call_op = EmitC_CallOpaqueOp(
                op_name,
                [A, C],
                [C.type],
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                    ]
                ),
            )

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToDiv(RewritePattern):
    """
    Convert linalg.generic with arith.divf to emitc.call_opaque calling div_operator.

    Handles two cases:
    1. Tensor / Tensor (two inputs) -> div_operator(&in1, &in2, &out)
    2. Tensor / Constant (one input + constant) -> tensor_imm_operator fallback

    Note:
    - div_operator in current NPU SDK requires same-shape inputs.
    - Broadcast div (e.g. 1x10 / 1x1) should be lowered by a dedicated
      broadcast-aware pattern (not OPERATION_DIV on tensor_tensor_operator).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a simple element-wise operation
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return  # Not an element-wise operation

        inputs = op.inputs
        outputs = op.outputs

        # Check body contains divf + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check if first operation is divf
        # Need to check both the type name and the operation name
        div_op = body_ops[0]
        op_type_name = type(div_op).__name__.upper()

        # Check if this is a divf operation
        is_divf = "DIVF" in op_type_name

        # If not, check if it has a name attribute that contains "divf"
        if not is_divf and hasattr(div_op, "name"):
            op_name = str(div_op.name)
            is_divf = "divf" in op_name.lower() or "arith.divf" in op_name

        if not is_divf:
            return  # Not a division operation

        # Get the operands of divf
        # In linalg.generic body, operands are either block args (from inputs) or constants
        div_operands = div_op.operands
        if len(div_operands) != 2:
            return

        # Check if we have tensor/tensor or tensor/constant
        # Block args: %arg0, %arg1, etc. correspond to inputs
        lhs = div_operands[0]
        rhs = div_operands[1]

        # Check if operands are block arguments (from inputs) or constants
        lhs_is_input = isinstance(lhs, BlockArgument)
        rhs_is_input = isinstance(rhs, BlockArgument)

        if lhs_is_input and rhs_is_input:
            # Case 1: Tensor / Tensor
            if len(inputs) != 2 or len(outputs) != 1:
                return

            A = inputs[0]
            B = inputs[1]
            C = outputs[0]

            # Use dedicated SDK div operator instead of tensor_tensor_operator(OPERATION_DIV).
            # This matches current SDK API surface and avoids unsupported op enum paths.
            call_op = EmitC_CallOpaqueOp(
                "div_operator",
                [A, B, C],
                [IntegerType(32)],  # Returns int status
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                    ]
                ),
            )
            rewriter.insert_op_before_matched_op(call_op)
            rewriter.erase_matched_op()
        elif (lhs_is_input and not rhs_is_input) or (not lhs_is_input and rhs_is_input):
            # Case 2: Tensor / Constant or Constant / Tensor
            if len(inputs) != 1 or len(outputs) != 1:
                return

            # Determine which side is the input tensor
            if lhs_is_input:
                tensor_input = inputs[0]
                const_op = rhs
                is_rhs = True
            else:
                tensor_input = inputs[0]
                const_op = lhs
                is_rhs = False

            # Get the operation that produced this result
            if isinstance(const_op, OpResult):
                const_op = const_op.op

            # Check if constant is actually an arith.constant or emitc_ext.constant
            is_constant = isinstance(const_op, arith.ConstantOp) or isinstance(
                const_op, emitc.EmitC_ConstantOp
            )

            # Also check by type name
            if not is_constant:
                const_op_type_name = type(const_op).__name__.upper()
                is_constant = "CONSTANT" in const_op_type_name

            if not is_constant:
                return  # Only handle simple constants for now

            # Get constant value
            if hasattr(const_op, "value"):
                const_value = const_op.value
            elif hasattr(const_op, "attributes"):
                # Try to get value from attributes
                const_value_attr = const_op.attributes.get("value")
                if const_value_attr:
                    const_value = const_value_attr.data
                else:
                    return
            else:
                return

            const_value_str = str(const_value)
            # Clean up the constant value string (remove type annotation like " : f32")
            if ":" in const_value_str:
                const_value_str = const_value_str.split(":")[0].strip()

            C = outputs[0]

            # For division by constant, use tensor_imm_operator
            # This handles tensor / scalar operations efficiently
            # We need to pass: tensor_in, tensor_out, imm_val, imm_wd, imm_type, tensor_op
            # For now, create an emitc.call_opaque with these parameters
            # Parameters will be: tensor_input, C, constant_value, WIDTH_8, TYPE_FP, OPERATION_DIV

            # Create constant for the immediate value (as float)
            # The constant value needs to be passed as a float value
            imm_val_const = EmitCConstantOp(const_op.value)
            rewriter.insert_op_before_matched_op(imm_val_const)

            # Create constants for WIDTH_8, TYPE_FP, OPERATION_DIV
            width_8_const = EmitCConstantOp(
                IntegerAttr(8, IntegerType(32))
            )  # WIDTH_8 = 8
            type_fp_const = EmitCConstantOp(
                IntegerAttr(1, IntegerType(32))
            )  # TYPE_FP = 1 (assuming)
            operation_div_const = EmitCConstantOp(
                IntegerAttr(41, IntegerType(32))
            )  # OPERATION_DIV = 41

            rewriter.insert_op_before_matched_op(width_8_const)
            rewriter.insert_op_before_matched_op(type_fp_const)
            rewriter.insert_op_before_matched_op(operation_div_const)

            # Create tensor_imm_operator call
            call_op = EmitC_CallOpaqueOp(
                "tensor_imm_operator",
                [
                    tensor_input,
                    C,
                    imm_val_const.results[0],
                    width_8_const.results[0],
                    type_fp_const.results[0],
                    operation_div_const.results[0],
                ],
                [IntegerType(32)],  # Returns int status
                args=ArrayAttr(
                    [
                        StringAttr("Tensor*"),
                        StringAttr("Tensor*"),
                        StringAttr("float"),
                        StringAttr("uint32_t"),
                        StringAttr("uint32_t"),
                        StringAttr("uint32_t"),
                    ]
                ),
            )
            rewriter.insert_op_before_matched_op(call_op)
            rewriter.erase_matched_op()
        else:
            # Both are constants - this should be optimized away, not handled here
            return


class ConvertLinalgGenericToMul(RewritePattern):
    """Convert linalg.generic with arith.mulf to emitc.call_opaque calling tensor_tensor_operator with OPERATION_MUL"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 2 or len(outputs) != 1:
            return

        # Check body contains only arith.mulf
        body_ops = list(op.body.block.ops)
        if len(body_ops) != 2:
            return

        if not isinstance(body_ops[0], arith.MulfOp):
            return

        A = inputs[0]
        B = inputs[1]
        C = outputs[0]

        # Create constant for OPERATION_MUL (value should match enum definition)
        # TODO: Adjust this value based on the actual enum definition in the header file
        c_op_type = EmitCConstantOp(IntegerAttr(40, IntegerType(32)))
        rewriter.insert_op_before_matched_op(c_op_type)

        call_op = EmitC_CallOpaqueOp(
            "tensor_tensor_operator",
            [A, B, C, c_op_type.results[0]],
            [IntegerType(32)],  # Returns int status
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("uint32_t"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToBnVectorEmitC(RewritePattern):
    """
    Lower fused batch-norm style linalg.generic (5 inputs, 1 output) to three
    tensor_vector_operator calls matching NPU_SDK test_case_i8_conv_convert.c:
    SUB vs running mean, MUL vs gamma (scale), ADD vs beta.

    This replaces the full per-element math (variance / rsqrt) with the same
    simplified NPU vector chain used in the handwritten INT8 test.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        # MLIR may print all memref operands in one list; xDSL may expose them
        # as either (5 inputs, 1 output) or (6 inputs, 0 outputs) depending on
        # how operandSegmentSizes was parsed.
        flat_ops = tuple(op.inputs) + tuple(op.outputs)
        body_ops = list(op.body.block.ops)
        if len(flat_ops) != 6 or len(body_ops) < 8:
            return

        if body_ops[0].name != "arith.truncf":
            return

        # linalg-generic-to-emitc runs before arith-to-emitc: BN may still use
        # arith.addf / arith.mulf, or already lowered emitc.add / emitc.mul.
        def _is_bin_add(o: Operation) -> bool:
            return o.name in ("arith.addf", "emitc.add")

        def _is_bin_mul(o: Operation) -> bool:
            return o.name in ("arith.mulf", "emitc.mul")

        if not _is_bin_add(body_ops[1]):
            return
        if body_ops[2].name != "math.rsqrt":
            return
        if body_ops[3].name != "arith.subf":
            return
        if not (_is_bin_mul(body_ops[4]) and _is_bin_mul(body_ops[5])):
            return
        if not _is_bin_add(body_ops[6]):
            return
        if body_ops[-1].name != "linalg.yield":
            return

        conv_or_x = flat_ops[0]
        gamma = flat_ops[1]
        beta = flat_ops[2]
        mean = flat_ops[3]
        out = flat_ops[5]

        c_sub = EmitCConstantOp(IntegerAttr(1, IntegerType(32)))
        c_mul = EmitCConstantOp(IntegerAttr(40, IntegerType(32)))
        c_add = EmitCConstantOp(IntegerAttr(0, IntegerType(32)))

        call_sub = EmitC_CallOpaqueOp(
            "tensor_vector_operator",
            [conv_or_x, mean, out, c_sub.results[0]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("uint32_t"),
                ]
            ),
        )
        call_mul = EmitC_CallOpaqueOp(
            "tensor_vector_operator",
            [out, gamma, out, c_mul.results[0]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("uint32_t"),
                ]
            ),
        )
        call_add = EmitC_CallOpaqueOp(
            "tensor_vector_operator",
            [out, beta, out, c_add.results[0]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("uint32_t"),
                ]
            ),
        )

        # insert_op_before_matched_op stacks before the generic; insert bottom-up
        # so each constant sits immediately above its call (SUB, then MUL, then ADD).
        for o in (c_sub, call_sub, c_mul, call_mul, c_add, call_add):
            rewriter.insert_op_before_matched_op(o)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToReluEmitC(RewritePattern):
    """Lower linalg.generic ReLU (cmpf + select + yield) to relu_operator."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs
        if len(inputs) != 1 or len(outputs) != 1:
            return

        body_ops = list(op.body.block.ops)
        if len(body_ops) != 3:
            return

        if not isinstance(body_ops[0], arith.CmpfOp):
            return
        if not isinstance(body_ops[1], arith.SelectOp):
            return
        if body_ops[2].name != "linalg.yield":
            return

        inp = inputs[0]
        outp = outputs[0]

        call_op = EmitC_CallOpaqueOp(
            "relu_operator",
            [inp, outp],
            [IntegerType(32)],
            args=ArrayAttr([StringAttr("Tensor*"), StringAttr("Tensor*")]),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToSquare(RewritePattern):
    """
    Convert linalg.generic with math.fpowi(exponent=2) to emitc.call_opaque calling square_operator
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check iterator types - must be all parallel
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        # Must have 1 input, 1 output
        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains math.fpowi + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check for fpowi operation - be more flexible with matching
        fpowi_op = body_ops[0]

        # Check operation name contains "Fpowi" or "fpowi"
        op_type_name = type(fpowi_op).__name__
        # Convert to uppercase for case-insensitive matching
        is_fpowi = "FPOWI" in op_type_name.upper() or "IPOWI" in op_type_name.upper()

        # Also check if it's a math operation by checking dialect
        if not is_fpowi and hasattr(fpowi_op, "name"):
            op_name_str = str(fpowi_op.name)
            is_fpowi = (
                "fpowi" in op_name_str.lower() or "math.fpowi" in op_name_str.lower()
            )

        if not is_fpowi:
            return  # Not a fpowi operation

        # Get operands
        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        # because linalg.generic has 0 results but emitc.call_opaque returns 1 result
        call_op = EmitC_CallOpaqueOp(
            "square_operator",
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToExp(RewritePattern):
    """Convert linalg.generic with math.exp to emitc.call_opaque calling lut_exp"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains math.exp + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) != 2:
            return

        if not isinstance(body_ops[0], math.ExpOp):
            return

        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        call_op = EmitC_CallOpaqueOp(
            "lut_exp",
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToRsqrt(RewritePattern):
    """
    Convert linalg.generic with math.rsqrt to emitc.call_opaque calling lut_squareroot
    Note: NPU has squareroot, rsqrt = 1/sqrt, need to handle this separately
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        iterator_types = op.iterator_types.data
        if not all(it.data == "parallel" for it in iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains math.rsqrt + linalg.yield
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check for rsqrt operation - be more flexible with matching
        rsqrt_op = body_ops[0]

        # Check operation name contains "Rsqrt" or "rsqrt"
        op_type_name = type(rsqrt_op).__name__
        is_rsqrt = "RSQRT" in op_type_name.upper() or "SQRT" in op_type_name.upper()

        # Also check if it's a math operation by checking dialect
        if not is_rsqrt and hasattr(rsqrt_op, "name"):
            op_name_str = str(rsqrt_op.name)
            is_rsqrt = (
                "rsqrt" in op_name_str.lower() or "math.rsqrt" in op_name_str.lower()
            )

        # Check if it's actually a RsqrtOp instance
        if not is_rsqrt and isinstance(rsqrt_op, math.RsqrtOp):
            is_rsqrt = True

        if not is_rsqrt:
            return  # Not a rsqrt operation

        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        call_op = EmitC_CallOpaqueOp(
            "lut_squareroot",
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToReduceSum(RewritePattern):
    """
    Convert linalg.generic reduction with arith.addf to emitc.call_opaque calling reduce_dimN_sum

    Input:
      linalg.generic {indexing_maps = [#map, #map1],
                     iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%A : memref<BxMxKxf32>) outs(%C : memref<BxMx1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.addf %in, %out : f32
          linalg.yield %6 : f32
      }

    Output:
      emitc.call_opaque ``reduce_dim{N}_sum`` where *N* is the NPU ``Tensor``
      axis (aligned with ``linalg_mlir_to_c._npu_view_dims`` / ``make_tensor``).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a reduction operation
        iterator_types = op.iterator_types.data

        # Should have at least one reduction iterator
        if len(iterator_types) < 1:
            return

        # Count reduction iterators
        reduction_count = sum(1 for it in iterator_types if it.data == "reduction")
        if reduction_count == 0:
            return

        # All others should be parallel
        parallel_count = sum(1 for it in iterator_types if it.data == "parallel")
        if parallel_count + reduction_count != len(iterator_types):
            return  # Mixed or unknown iterator types

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains arith.addf OR emitc.add (if already converted by arith_to_emitc)
        body_ops = list(op.body.block.ops)
        if len(body_ops) < 2:
            return

        # Check if first operation is addf or emitc.add
        add_op = body_ops[0]
        is_add = isinstance(add_op, arith.AddfOp)

        # If not arith.AddfOp, check if it's emitc.add
        if not is_add:
            op_type_name = type(add_op).__name__
            is_add = (
                "ADD" in op_type_name.upper() or "emitc.add" in str(add_op.name).lower()
            )

        if not is_add:
            return

        # Get input shape
        input_type = inputs[0].type
        if not isinstance(input_type, MemRefType):
            return

        shape = input_type.shape.data
        num_dims = len(shape)

        # Determine reduction dimension based on output shape
        output_type = outputs[0].type
        if not isinstance(output_type, MemRefType):
            return

        output_shape = output_type.shape.data

        # Find which dimensions were reduced
        # A dimension is reduced if input size > 1 and output size == 1
        reduced_dims = []
        for i in range(num_dims):
            if i < len(output_shape):
                input_dim = shape[i].data
                output_dim = output_shape[i].data
                # Only count as reduced if input > 1 and output == 1
                if input_dim > 1 and output_dim == 1:
                    reduced_dims.append(i)
            elif shape[i].data > 1:
                # Dimension is missing from output (completely reduced)
                reduced_dims.append(i)

        # Generate operator name based on reduced dimensions
        if len(reduced_dims) == 1:
            dim_num = _memref_axis_to_npu_sdk_reduce_dim(reduced_dims[0], num_dims)
            op_name = f"reduce_dim{dim_num}_sum"
        elif len(reduced_dims) == 2:
            # Memref axis indices (pooling); SDK only exposes reduce_dim2_dim1_*.
            dim_nums = sorted(reduced_dims, reverse=True)
            op_name = f"reduce_dim{dim_nums[0]}_dim{dim_nums[1]}_sum"
        else:
            # Fallback
            op_name = "reduce_dim2_sum"  # Most common case

        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        call_op = EmitC_CallOpaqueOp(
            op_name,
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToReduceMax(RewritePattern):
    """
    Convert linalg.generic reduction with arith.maximumf to emitc.call_opaque calling reduce_dimN_max

    Input:
      linalg.generic {indexing_maps = [#map, #map1],
                     iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%A : memref<BxMxKxf32>) outs(%C : memref<BxMx1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.maximumf %in, %out : f32
          linalg.yield %6 : f32
      }

    Output:
      emitc.call_opaque ``reduce_dim{N}_max`` with *N* = NPU Tensor axis (see
      ``_memref_axis_to_npu_sdk_reduce_dim``).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a reduction operation
        iterator_types = op.iterator_types.data

        # Should have at least one reduction iterator
        if len(iterator_types) < 1:
            return

        # Count reduction iterators
        reduction_count = sum(1 for it in iterator_types if it.data == "reduction")
        if reduction_count == 0:
            return

        # All others should be parallel
        parallel_count = sum(1 for it in iterator_types if it.data == "parallel")
        if parallel_count + reduction_count != len(iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains arith.maximumf
        body_ops = list(op.body.block.ops)
        if len(body_ops) != 2:
            return

        if not isinstance(body_ops[0], arith.MaximumfOp):
            return

        input_type = inputs[0].type
        if not isinstance(input_type, MemRefType):
            return

        shape = input_type.shape.data
        num_dims = len(shape)

        output_type = outputs[0].type
        if not isinstance(output_type, MemRefType):
            return

        output_shape = output_type.shape.data

        # Find which dimensions were reduced
        # A dimension is reduced if input size > 1 and output size == 1
        reduced_dims = []
        for i in range(num_dims):
            if i < len(output_shape):
                input_dim = shape[i].data
                output_dim = output_shape[i].data
                # Only count as reduced if input > 1 and output == 1
                if input_dim > 1 and output_dim == 1:
                    reduced_dims.append(i)
            elif shape[i].data > 1:
                # Dimension is missing from output (completely reduced)
                reduced_dims.append(i)

        # Generate operator name based on reduced dimensions
        if len(reduced_dims) == 1:
            dim_num = _memref_axis_to_npu_sdk_reduce_dim(reduced_dims[0], num_dims)
            op_name = f"reduce_dim{dim_num}_max"
        elif len(reduced_dims) == 2:
            dim_nums = sorted(reduced_dims, reverse=True)
            op_name = f"reduce_dim{dim_nums[0]}_dim{dim_nums[1]}_max"
        else:
            return

        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        call_op = EmitC_CallOpaqueOp(
            op_name,
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgGenericToReduceMin(RewritePattern):
    """
    Convert linalg.generic reduction with arith.minimumf to emitc.call_opaque calling reduce_dimN_min

    Input:
      linalg.generic {indexing_maps = [#map, #map1],
                     iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%A : memref<BxMxKxf32>) outs(%C : memref<BxMx1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.minimumf %in, %out : f32
          linalg.yield %6 : f32
      }

    Output:
      emitc.call_opaque ``reduce_dim{N}_min`` with *N* = NPU Tensor axis (see
      ``_memref_axis_to_npu_sdk_reduce_dim``).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Check if this is a reduction operation
        iterator_types = op.iterator_types.data

        # Should have at least one reduction iterator
        if len(iterator_types) < 1:
            return

        # Count reduction iterators
        reduction_count = sum(1 for it in iterator_types if it.data == "reduction")
        if reduction_count == 0:
            return

        # All others should be parallel
        parallel_count = sum(1 for it in iterator_types if it.data == "parallel")
        if parallel_count + reduction_count != len(iterator_types):
            return

        inputs = op.inputs
        outputs = op.outputs

        if len(inputs) != 1 or len(outputs) != 1:
            return

        # Check body contains arith.minimumf
        body_ops = list(op.body.block.ops)
        if len(body_ops) != 2:
            return

        if not isinstance(body_ops[0], arith.MinimumfOp):
            return

        input_type = inputs[0].type
        if not isinstance(input_type, MemRefType):
            return

        shape = input_type.shape.data
        num_dims = len(shape)

        output_type = outputs[0].type
        if not isinstance(output_type, MemRefType):
            return

        output_shape = output_type.shape.data

        # Find which dimensions were reduced
        # A dimension is reduced if input size > 1 and output size == 1
        reduced_dims = []
        for i in range(num_dims):
            if i < len(output_shape):
                input_dim = shape[i].data
                output_dim = output_shape[i].data
                # Only count as reduced if input > 1 and output == 1
                if input_dim > 1 and output_dim == 1:
                    reduced_dims.append(i)
            elif shape[i].data > 1:
                # Dimension is missing from output (completely reduced)
                reduced_dims.append(i)

        # Generate operator name based on reduced dimensions
        if len(reduced_dims) == 1:
            dim_num = _memref_axis_to_npu_sdk_reduce_dim(reduced_dims[0], num_dims)
            op_name = f"reduce_dim{dim_num}_min"
        elif len(reduced_dims) == 2:
            dim_nums = sorted(reduced_dims, reverse=True)
            op_name = f"reduce_dim{dim_nums[0]}_dim{dim_nums[1]}_min"
        else:
            return

        A = inputs[0]
        C = outputs[0]

        # Use insert_op_before_matched_op + erase_matched_op instead of replace_matched_op
        call_op = EmitC_CallOpaqueOp(
            op_name,
            [A, C],
            [C.type],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgConv2DNchwFchwToEmitC(RewritePattern):
    """Convert linalg.conv_2d_nchw_fchw to emitc.call_opaque(conv_operator).

    The C backend (``linalg_mlir_to_c``) lowers this like NPU_SDK
    ``test_case_i8_conv_convert.c`` / ``conv_operator_call_example.c``:
    ``conv_operator(&shifted, &out, &out, &conv_opt)`` with per-``kx`` slices,
    and ``make_tensor`` shapes (C,H,W) for N=1 NCHW activations and
    (cout,cin,ky*kx) for FCHW weights.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.name != "linalg.conv_2d_nchw_fchw":
            return
        if len(op.operands) < 3:
            return

        tensor_in = op.operands[0]
        tensor_w = op.operands[1]
        tensor_out = op.operands[2]

        call_op = EmitC_CallOpaqueOp(
            "conv_operator",
            [tensor_in, tensor_w, tensor_out],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgPoolingNchwSumToEmitC(RewritePattern):
    """Convert linalg.pooling_nchw_sum to emitc.call_opaque(pooling_nchw_sum)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.name != "linalg.pooling_nchw_sum":
            return
        if len(op.operands) < 3:
            return

        call_op = EmitC_CallOpaqueOp(
            "pooling_nchw_sum",
            [op.operands[0], op.operands[1], op.operands[2]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgMatmulToEmitC(RewritePattern):
    """Convert linalg.matmul to emitc.call_opaque(matmul_operator)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.name != "linalg.matmul":
            return
        if len(op.operands) < 3:
            return

        call_op = EmitC_CallOpaqueOp(
            "matmul_operator",
            [op.operands[0], op.operands[1], op.operands[2]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertLinalgTransposeToEmitC(RewritePattern):
    """Convert linalg.transpose to emitc.call_opaque(transpose_operator)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.name != "linalg.transpose":
            return
        if len(op.operands) < 2:
            return

        dim_axis = EmitCConstantOp(IntegerAttr(0, IntegerType(32)))
        rewriter.insert_op_before_matched_op(dim_axis)

        call_op = EmitC_CallOpaqueOp(
            "transpose_operator",
            [op.operands[0], op.operands[1], dim_axis.results[0]],
            [IntegerType(32)],
            args=ArrayAttr(
                [
                    StringAttr("Tensor*"),
                    StringAttr("Tensor*"),
                    StringAttr("int"),
                ]
            ),
        )
        rewriter.insert_op_before_matched_op(call_op)
        rewriter.erase_matched_op()


class ConvertMemrefCollapseShapeToEmitC(RewritePattern):
    """Convert memref.collapse_shape to emitc.call_opaque(flatten_view_operator)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if op.name != "memref.collapse_shape":
            return
        if len(op.operands) < 1 or len(op.results) != 1:
            return

        call_op = EmitC_CallOpaqueOp(
            "flatten_view_operator",
            [op.operands[0]],
            [op.results[0].type],
            args=ArrayAttr([StringAttr("Tensor*")]),
        )
        rewriter.replace_matched_op(call_op)


# class ConvertRMSNormSequenceToOperator(RewritePattern):
#     """
#     Identify and optimize RMSNorm sequence to single rmsnorm_operator call

#     RMSNorm Pattern:
#       1. Square: x^2
#       2. Reduce Sum: sum(x^2, axis=-1)
#       3. Divide by dim: sum / dim_size
#       4. Add epsilon: mean + eps
#       5. Rsqrt: 1 / sqrt(...)
#       6. Multiply by input: x * rsqrt
#       7. Multiply by gamma: result * gamma

#     When this pattern is detected, replace with:
#       rmsnorm_operator(x, gamma, output, epsilon)

#     Note: This pattern is applied at function level to detect the sequence.
#     If the full sequence is not found, individual ops will be converted
#     by other patterns (fallback to basic operators).
#     """

#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
#         # Only process functions, not individual operations
#         if not isinstance(op, func.FuncOp):
#             return

#         import sys
#         print(f"[DEBUG] Processing function: {op.sym_name.data}", file=sys.stderr)

#         # Get all operations in the function body, but filter only linalg.generic ops
#         all_ops = list(op.body.ops)
#         linalg_ops = [o for o in all_ops if isinstance(o, linalg.GenericOp)]

#         print(f"[DEBUG] Found {len(linalg_ops)} linalg.generic operations", file=sys.stderr)

#         if len(linalg_ops) < 7:
#             print(f"[DEBUG] Not enough linalg.generic operations for RMSNorm", file=sys.stderr)
#             return  # Not enough linalg.generic operations for RMSNorm

#         # Try to find RMSNorm pattern in linalg.generic operations
#         rmsnorm_candidates = []

#         for i in range(len(linalg_ops) - 6):
#             # Check 7-operation window (only linalg.generic ops)
#             window = linalg_ops[i:i+7]

#             if self._is_rmsnorm_sequence(window):
#                 print(f"[DEBUG] Found RMSNorm sequence at index {i}", file=sys.stderr)
#                 # Extract the actual operations and their dataflow
#                 sequence_info = self._extract_rmsnorm_sequence_info(window)
#                 if sequence_info:
#                     print(f"[DEBUG] Successfully extracted RMSNorm sequence info", file=sys.stderr)
#                     rmsnorm_candidates.append((i, window, sequence_info))
#                 else:
#                     print(f"[DEBUG] Failed to extract sequence info", file=sys.stderr)

#         print(f"[DEBUG] Found {len(rmsnorm_candidates)} RMSNorm candidates", file=sys.stderr)

#         # If RMSNorm pattern found, replace with single operator call
#         if rmsnorm_candidates:
#             # Replace the first found sequence
#             start_idx, window, seq_info = rmsnorm_candidates[0]

#             # Get input, gamma, output, and epsilon
#             input_tensor = seq_info['input']
#             gamma_tensor = seq_info['gamma']
#             output_tensor = seq_info['output']
#             epsilon_value = seq_info['epsilon']

#             # Get pointers for inputs (handle memref to pointer conversion)
#             try:
#                 # For memref inputs, we need to use them directly in emitc.call_opaque
#                 # The memref-to-emitc pass will handle pointer conversion later
#                 ptr_input = input_tensor
#                 ptr_gamma = gamma_tensor
#                 ptr_output = output_tensor
#             except:
#                 # If pointer conversion fails, skip this optimization
#                 return

#             # Create rmsnorm_operator call
#             # Signature: rmsnorm_operator(Tensor* input, Tensor* gamma, Tensor* output, float epsilon)
#             rmsnorm_call = emitc.EmitC_CallOpaqueOp(
#                 callee="rmsnorm_operator",
#                 call_args=[ptr_input, ptr_gamma, ptr_output],
#                 result_types=[],
#                 args=ArrayAttr([
#                     StringAttr("Tensor*"),
#                     StringAttr("Tensor*"),
#                     StringAttr("Tensor*"),
#                 ]),
#             )

#             # Replace all 7 operations with the single call
#             # For now, just insert before the matched operation and don't erase
#             # This helps us debug the insertion logic
#             rewriter.insert_op_before_matched_op(rmsnorm_call)

#             # TODO: Erase the 7 operations - commented out for debugging
#             # for op_in_window in reversed(window):
#             #     rewriter.erase_op(op_in_window)

#     def _extract_rmsnorm_sequence_info(self, ops):
#         """Extract input, gamma, output, and epsilon from RMSNorm sequence"""
#         if len(ops) != 7:
#             return None

#         try:
#             # op[0]: square operation - takes input
#             square_op = ops[0]
#             if not isinstance(square_op, linalg.GenericOp):
#                 return None
#             input_tensor = square_op.inputs[0]

#             # op[1]: reduce sum - produces intermediate
#             # op[2]: divide by dim - produces mean
#             # op[3]: add epsilon - produces mean_eps
#             # op[4]: rsqrt - produces rsqrt_val

#             # op[5]: multiply (input * rsqrt)
#             mul1_op = ops[5]
#             if not isinstance(mul1_op, linalg.GenericOp):
#                 return None

#             # op[6]: multiply (result * gamma) - produces final output
#             mul2_op = ops[6]
#             if not isinstance(mul2_op, linalg.GenericOp):
#                 return None
#             gamma_tensor = mul2_op.inputs[0]
#             output_tensor = mul2_op.outputs[0]

#             # Extract epsilon from op[3] (add operation)
#             add_op = ops[3]
#             if not isinstance(add_op, linalg.GenericOp):
#                 return None

#             # Get epsilon value (second input to add)
#             epsilon_input = add_op.inputs[1]

#             # Check if epsilon is a constant or memref global
#             epsilon_value = None
#             # For now, use default epsilon value since extraction from DenseElementsAttr is complex
#             # In production, you would extract the actual value from the DenseIntOrFPElementsAttr
#             epsilon_value = 1.0e-5

#             if epsilon_value is None:
#                 # Default epsilon value
#                 epsilon_value = 1.0e-5

#             return {
#                 'input': input_tensor,
#                 'gamma': gamma_tensor,
#                 'output': output_tensor,
#                 'epsilon': epsilon_value,
#             }

#         except (IndexError, AttributeError, TypeError) as e:
#             import sys
#             print(f"[DEBUG] _extract_rmsnorm_sequence_info error: {e}", file=sys.stderr)
#             return None

#     def _is_rmsnorm_sequence(self, ops):
#         """Check if a sequence of operations matches RMSNorm pattern"""
#         import sys
#         if len(ops) != 7:
#             print(f"[DEBUG] _is_rmsnorm_sequence: Wrong length {len(ops)}", file=sys.stderr)
#             return False

#         # Simplified check: verify operation types
#         checks = [
#             ("square", self._is_square_op(ops[0])),
#             ("reduce_sum", self._is_reduce_sum_op(ops[1])),
#             ("div", self._is_div_op(ops[2])),
#             ("add", self._is_add_op(ops[3])),
#             ("rsqrt", self._is_rsqrt_op(ops[4])),
#             ("mul1", self._is_mul_op(ops[5])),
#             ("mul2", self._is_mul_op(ops[6])),
#         ]

#         for name, result in checks:
#             print(f"[DEBUG] {name}: {result}", file=sys.stderr)

#         return all(check for _, check in checks)

#     def _is_square_op(self, op):
#         """Check if operation computes square"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         body_ops = list(op.body.block.ops)
#         return len(body_ops) == 2 and "FPOWI" in type(body_ops[0]).__name__.upper()

#     def _is_reduce_sum_op(self, op):
#         """Check if operation is reduction sum"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         iterator_types = op.iterator_types.data
#         has_reduction = any(it.data == "reduction" for it in iterator_types)
#         body_ops = list(op.body.block.ops)
#         has_addf = len(body_ops) == 2 and isinstance(body_ops[0], arith.AddfOp)
#         return has_reduction and has_addf

#     def _is_div_op(self, op):
#         """Check if operation is division"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         body_ops = list(op.body.block.ops)
#         return len(body_ops) == 2 and "DIVF" in type(body_ops[0]).__name__.upper()

#     def _is_add_op(self, op):
#         """Check if operation is addition (epsilon add)"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         body_ops = list(op.body.block.ops)
#         # Allow 2 or 3 operations: (addf, yield) or (truncf, addf, yield)
#         if len(body_ops) < 2 or len(body_ops) > 3:
#             return False
#         # Check if any of the operations is addf
#         return any(isinstance(op, arith.AddfOp) for op in body_ops)

#     def _is_rsqrt_op(self, op):
#         """Check if operation is rsqrt"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         body_ops = list(op.body.block.ops)
#         return len(body_ops) == 2 and isinstance(body_ops[0], math.RsqrtOp)

#     def _is_mul_op(self, op):
#         """Check if operation is multiplication"""
#         if not isinstance(op, linalg.GenericOp):
#             return False
#         body_ops = list(op.body.block.ops)
#         return len(body_ops) == 2 and isinstance(body_ops[0], arith.MulfOp)


class RMSNormOptimizationPass(ModulePass):
    """
    Dedicated pass to optimize RMSNorm sequences
    This pass runs before linalg-generic-to-emitc to detect and optimize RMSNorm patterns
    """

    name = "rmsnorm-optimization"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # Apply RMSNorm pattern matching
        pattern = ConvertRMSNormSequenceToOperator()
        # Use apply_recursively=True to process all functions in the module
        try:
            PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)
        except Exception as e:
            # If RMSNorm optimization fails, continue anyway
            # This is an optional optimization pass
            import sys

            print(f"[DEBUG] RMSNorm optimization: {e}", file=sys.stderr)


class LinalgGenericToEmitCPass(ModulePass):
    """
    Pass to convert all linalg.generic operations to emitc.call_opaque

    This pass applies multiple patterns to convert different linalg.generic
    patterns to their corresponding NPU SDK operator calls.
    """

    name = "linalg-generic-to-emitc"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        # Apply all conversion patterns
        patterns = [
            ConvertLinalgConv2DNchwFchwToEmitC(),
            ConvertLinalgPoolingNchwSumToEmitC(),
            ConvertLinalgMatmulToEmitC(),
            ConvertLinalgTransposeToEmitC(),
            ConvertMemrefCollapseShapeToEmitC(),
            # BN / ReLU must run before generic binary patterns and identity.
            ConvertLinalgGenericToBnVectorEmitC(),
            ConvertLinalgGenericToReluEmitC(),
            ConvertLinalgGenericToIdentity(),  # Copy / broadcast (single yield)
            ConvertLinalgGenericToAdd(),
            ConvertLinalgGenericToSub(),
            ConvertLinalgGenericToMul(),
            ConvertLinalgGenericToDiv(),
            ConvertLinalgGenericToSquare(),
            ConvertLinalgGenericToExp(),
            ConvertLinalgGenericToRsqrt(),
            ConvertLinalgGenericToReduceSum(),
            ConvertLinalgGenericToReduceMax(),
            ConvertLinalgGenericToReduceMin(),
            # Note: RMSNorm optimization is handled by a separate pass (RMSNormOptimizationPass)
        ]

        # Apply patterns with rewriter
        for pattern in patterns:
            try:
                PatternRewriteWalker(pattern, apply_recursively=True).rewrite_module(op)
            except Exception:
                # Continue with other patterns if one fails
                pass
