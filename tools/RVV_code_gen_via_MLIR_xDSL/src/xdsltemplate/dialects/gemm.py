from xdsl.dialects.builtin import (
    IndexType,
)
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class AcolOp(IRDLOperation):
    name = "gemm.acol"

    i = operand_def(IndexType)
    j = operand_def(IndexType)
    lda = operand_def(IndexType)
    res = result_def(IndexType)

    def __init__(self, i: SSAValue, j: SSAValue, lda: SSAValue):
        super().__init__(
            operands=[i, j, lda],
            result_types=[IndexType()],
        )

    # def print(self, printer: Printer):
    #     printer.print_string(" ")
    #     printer.print_operand(self.i)
    #     printer.print_string(", ")
    #     printer.print_operand(self.j)
    #     printer.print_string(", ")
    #     printer.print_operand(self.lda)
    #     printer.print_string(" : ")
    #     printer.print_operation_type()
    #
    # @classmethod
    # def parse(cls, parser: Parser):
    #     i = parser.parse_operand()
    #     parser.parse_punctuation(",")
    #     j = parser.parse_operand()
    #     parser.parse_punctuation(",")
    #     lda = parser.parse_operand()
    #     parser.parse_punctuation(":")
    #     parser.parse_operation_type()
    #     return cls(i, j, lda)


@irdl_op_definition
class BcolOp(IRDLOperation):
    name = "gemm.bcol"

    i = operand_def(IndexType)
    j = operand_def(IndexType)
    ldb = operand_def(IndexType)
    res = result_def(IndexType)

    def __init__(self, i: SSAValue, j: SSAValue, ldb: SSAValue):
        super().__init__(
            operands=[i, j, ldb],
            result_types=[IndexType()],
        )


@irdl_op_definition
class CcolOp(IRDLOperation):
    name = "gemm.ccol"

    i = operand_def(IndexType)
    j = operand_def(IndexType)
    ldc = operand_def(IndexType)
    res = result_def(IndexType)

    def __init__(self, i: SSAValue, j: SSAValue, ldc: SSAValue):
        super().__init__(
            operands=[i, j, ldc],
            result_types=[IndexType()],
        )


GEMM = Dialect(
    "gemm",
    [
        AcolOp,
        BcolOp,
        CcolOp,
    ],
    [],
)
