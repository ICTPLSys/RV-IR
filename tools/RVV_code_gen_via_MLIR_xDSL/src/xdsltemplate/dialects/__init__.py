from collections.abc import Callable

from xdsl.ir import Dialect

from . import emitc_ext, emitc_ptr, gemm, hello, riscv, riscv_bpi, rvv


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    # Add your dialects here to be discovered by `xdsl-opt`
    return {
        "emitc_ext": lambda: emitc_ext.EmitC_Ext,
        "emitc_ptr": lambda: emitc_ptr.EmitC_Ptr,
        "gemm": lambda: gemm.GEMM,
        "hello": lambda: hello.Hello,
        "rair": lambda: riscv.RAIR,
        "riscv": lambda: riscv.RISCV,
        "riscv_bpi": lambda: riscv_bpi.RISCV_BPI,
        "rvv": lambda: rvv.RVV,
    }
