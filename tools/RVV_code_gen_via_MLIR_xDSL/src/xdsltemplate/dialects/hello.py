from xdsl.dialects.builtin import ModuleOp
from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class HelloWorldOp(IRDLOperation):
    name = "hello.world"

    def verify_(self) -> None:
        # Optional sanity check
        if len(self.operands) != 0:
            raise VerifyException("hello.world takes no operands")

    def __str__(self):
        return "hello.world"


def build_hello_module() -> ModuleOp:
    mod = ModuleOp([HelloWorldOp()])

    return mod
