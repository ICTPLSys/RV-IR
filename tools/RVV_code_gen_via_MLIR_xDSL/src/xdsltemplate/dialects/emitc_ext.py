# src/xdsltemplate/dialects/emitc_ext.py
from xdsl.dialects.builtin import Attribute, Data, IndexType
from xdsl.ir import Dialect, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
)


# ----------------------------------------------------------------------
# Custom EmitC type: !emitc.size_t
# ----------------------------------------------------------------------
@irdl_attr_definition
class EmitC_SizeTType(Data, TypeAttribute):
    name = "emitc.size_t"


@irdl_attr_definition
class EmitC_PtrType(Data, TypeAttribute):
    name = "emitc.ptr"


# ----------------------------------------------------------------------
# Custom EmitC op: emitc.constant
# ----------------------------------------------------------------------
# @irdl_op_definition
# class EmitCConstantOp(IRDLOperation):
#     """
#     Minimal EmitC constant operation.
#     Example:
#       %0 = emitc.constant {value = 0 : index} : !emitc.size_t
#     """
#     name = "emitc.constant"
#
#     value = opt_prop_def(Attribute)
#     res = result_def(EmitC_SizeTType)
#
#     def __init__(self, value: Attribute):
#         super().__init__(
#             attributes={"value": value},
#             result_types=[EmitC_SizeTType()],
#         )


@irdl_op_definition
class EmitCConstantOp(IRDLOperation):
    """
    Minimal EmitC constant operation.
    Example:
      %0 = emitc_ext.constant {value = 0 : index} : index
    """

    name = "emitc_ext.constant"

    value = opt_prop_def(Attribute)
    res = result_def(IndexType)

    def __init__(self, value: Attribute):
        super().__init__(
            attributes={"value": value},
            result_types=[value.type],
        )


#  Custom subscript type definiction
@irdl_op_definition
class EmitCSubscriptOp(IRDLOperation):
    name = "emitc.subscript"

    ptr = operand_def()
    idx = operand_def()
    result = result_def()

    def __init__(self, ptr, idx, result_type):
        super().__init__(operands=[ptr, idx], result_types=[result_type])


@irdl_op_definition
class EmitCLoadOp(IRDLOperation):
    name = "emitc.load"

    src = operand_def()
    result = result_def()

    def __init__(self, src, result_type):
        super().__init__(operands=[src], result_types=[result_type])


@irdl_op_definition
class EmitCMullOp(IRDLOperation):
    name = "emitc.mul"

    lhs = operand_def()
    rhs = operand_def()
    result = result_def()

    def __init__(self, lhs, rhs):
        result_type = lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


@irdl_op_definition
class EmitCAddOp(IRDLOperation):
    name = "emitc.add"

    lhs = operand_def()
    rhs = operand_def()
    result = result_def()

    def __init__(self, lhs, rhs):
        result_type = lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


@irdl_op_definition
class EmitCSubOp(IRDLOperation):
    name = "emitc.sub"

    lhs = operand_def()
    rhs = operand_def()
    result = result_def()

    def __init__(self, lhs, rhs):
        result_type = lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


@irdl_op_definition
class EmitCStoreOp(IRDLOperation):
    name = "emitc.store"

    lhs = operand_def()
    rhs = operand_def()
    result = result_def()

    def __init__(self, lhs, rhs):
        result_type = lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[result_type])


# @irdl_op_definition
# class EmitCAssignOp(IRDLOperation):
#     name = "emitc.assign"
#
#     src = operand_def()
#     result = result_def()
#
#     def __init__(self, src):
#         super().__init__(operands=[src], result_types=[result_type])


@irdl_op_definition
class EmitCAssignOp(IRDLOperation):
    name = "emitc.assign"

    value = operand_def()  # Value to assign
    dest = operand_def()  # Destination lvalue

    def __init__(self, value, dest):
        super().__init__(operands=[value, dest], result_types=[])


# Define the EmitC_Ext dialect with only EmitCConstantOp
EmitC_Ext = Dialect(
    "emitc_ext",  # Use a unique name to avoid conflict with standard emitc dialect
    [
        EmitCConstantOp,  # Only include the constant operation
    ],
    [],  # Don't include any types to avoid conflicts
)
