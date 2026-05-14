from collections.abc import Callable

from xdsl.passes import ModulePass

from . import (
    arith_to_emitc,
    gemm_to_arith,
    linalg_generic_to_emitc,
    memref_load_to_emitc,
    memref_store_to_emitc,
    memref_to_emitc,
    riscv_to_emitc,
    rvv_to_emitc,
    scf_to_emitc,
)


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """Return the list of all available passes."""

    # Add your passes here to be discovered by `xdsl-opt`
    return {
        "arith-to-emitc": lambda: arith_to_emitc.ArithToEmitCPass,
        "gemm-to-arith": lambda: gemm_to_arith.GemmToArithPass,
        "linalg-generic-to-emitc": lambda: linalg_generic_to_emitc.LinalgGenericToEmitCPass,
        "memref-load-to-emitc": lambda: memref_load_to_emitc.MemrefLoadToEmitcPass,
        "memref-store-to-emitc": lambda: memref_store_to_emitc.MemrefStoreToEmitcPass,
        "memref-to-emitc": lambda: memref_to_emitc.MemRefToEmitCPass,
        "rocc-to-emitc": lambda: riscv_to_emitc.RoCCToEmitCPass,
        "riscv-to-emitc": lambda: riscv_to_emitc.RoCCToEmitCPass,
        "rvv-to-emitc": lambda: rvv_to_emitc.RVVToEmitCPass,
        "scf-to-emitc": lambda: scf_to_emitc.SCFToEmitCPass,
    }
