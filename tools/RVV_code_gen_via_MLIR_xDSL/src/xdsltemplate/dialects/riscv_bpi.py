from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class RVBPIAddOp(IRDLOperation):
    name = "riscv_bpi.add"

    # Each operand must declare its type constraint
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    res = result_def(IntegerType(32))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(32)])


RiscvBPI = Dialect("riscv_bpi", [RVBPIAddOp], [])
