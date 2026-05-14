import argparse
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MLIR_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "mlir_output")
_DEFAULT_OUTPUT = os.path.join(_DEFAULT_MLIR_OUTPUT_DIR, "mnist_linalg.mlir")
_DEFAULT_MEMREF_OUTPUT = os.path.join(
    _DEFAULT_MLIR_OUTPUT_DIR, "mnist_linalg_memref.mlir"
)

# Keep the same torch-mlir python path setup style as other examples.
home_dir = os.path.expanduser("~")
# Lowest priority first: repeated sys.path.insert(0, ...) leaves the last path
# at the front, so the built torch-mlir package must be appended last here.
python_paths = [
    os.path.join(home_dir, "RV-IR/projects/pt1/python/torch_mlir"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/"),
    os.path.join(home_dir, "RV-IR/build/tools/torch-mlir/python_packages/torch_mlir"),
]
for path in python_paths:
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_paths + sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mlir import torchscript
from torch_mlir.ir import Location, Module, Operation
from torch_mlir.passmanager import PassManager


class MnistNet(nn.Module):
    """Small MNIST classifier compatible with linalg-on-tensors lowering."""

    def __init__(self):
        super().__init__()
        # Use a smaller hidden layer to reduce model size.
        self.fc1 = nn.Linear(14 * 14, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = x.reshape(-1, 14 * 14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


_BUFFERIZE_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries copy-before-write "
    "unknown-type-conversion=identity-layout-map},"
    "canonicalize,cse)"
)
_POST_CLEANUP_PIPELINE = "builtin.module(canonicalize,cse)"


def _mlir_operation(op) -> Operation:
    """Dialect OpView (e.g. memref.get_global) or raw Operation -> Operation."""
    if isinstance(op, Operation):
        return op
    return op.operation


def _symbol_string(attr) -> str:
    s = str(attr).replace('"', "").strip()
    if s.startswith("@"):
        s = s[1:]
    return s


def _iter_ops_depth_first(op):
    yield op
    mop = _mlir_operation(op)
    for region in mop.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _iter_ops_depth_first(child)


def _op_index_in_block(op):
    mop = _mlir_operation(op)
    for i, o in enumerate(mop.block.operations):
        if _mlir_operation(o) is mop:
            return i
    raise RuntimeError("op not found in its block")


def _op_block_order(op):
    mop = _mlir_operation(op)
    return (id(mop.block), _op_index_in_block(op))


def _fill_literal_for_type(elem_ty, fill_value: float) -> tuple[str, str]:
    """Return (literal, type_asm) for arith.constant in the snippet module."""
    et = str(elem_ty)
    if et == "f32":
        return (repr(float(fill_value)), "f32")
    if et == "f64":
        return (repr(float(fill_value)), "f64")
    if et in ("i32", "i64"):
        return ("0", et)
    # Best-effort: float literal with declared element type (may fail verify).
    return (repr(float(fill_value)), et)


def _bufferize_module(module: Module) -> None:
    with module.context:
        PassManager.parse(_BUFFERIZE_PIPELINE).run(module.operation)


def _post_cleanup_module(module: Module) -> None:
    with module.context:
        PassManager.parse(_POST_CLEANUP_PIPELINE).run(module.operation)


def replace_memref_globals_with_stack_alloc_fill(
    module: Module, fill_value: float = 1.0
) -> None:
    """Turn memref.global + memref.get_global into memref.alloc + linalg.fill.

    Matches the style used in RVV_code_gen_via_MLIR_xDSL riscv_tests (placeholder
    weights). Runs on an already bufferized module in-place.
    """
    module_op = module.operation
    ctx = module.context

    get_global_by_sym: dict[str, list] = defaultdict(list)
    for op in _iter_ops_depth_first(module_op):
        mop = _mlir_operation(op)
        if mop.name == "memref.get_global":
            sym = _symbol_string(mop.attributes["name"])
            get_global_by_sym[sym].append(op)

    with ctx, Location.unknown():
        for sym, gops in get_global_by_sym.items():
            first = min(gops, key=_op_block_order)
            first_m = _mlir_operation(first)
            memref_ty = first_m.results[0].type
            ty_str = str(memref_ty)
            lit, lit_ty = _fill_literal_for_type(memref_ty.element_type, fill_value)

            snippet = f"""
module {{
  func.func @__snippet_emit() {{
    %fv = arith.constant {lit} : {lit_ty}
    %a = memref.alloc() {{alignment = 64 : i64}} : {ty_str}
    linalg.fill ins(%fv : {lit_ty}) outs(%a : {ty_str})
    return
  }}
}}
"""
            tmp = Module.parse(snippet, ctx)
            tmp_body = tmp.operation.regions[0].blocks[0]
            tfunc = next(
                o
                for o in tmp_body.operations
                if _mlir_operation(o).name == "func.func"
            )
            tentry = _mlir_operation(tfunc).regions[0].blocks[0]
            tpl_ops = [
                o
                for o in tentry.operations
                if _mlir_operation(o).name != "func.return"
            ]
            for o in tpl_ops:
                o.move_before(first_m)
            alloc_op = tpl_ops[1]
            _mlir_operation(tfunc).erase()

            alloc_m = _mlir_operation(alloc_op)
            for g in gops:
                gm = _mlir_operation(g)
                gm.results[0].replace_all_uses_with(alloc_m.results[0])
                gm.erase()

    # Drop memref.global ops (constants are now stack placeholders).
    mod_body = module_op.regions[0].blocks[0]
    for op in list(mod_body.operations):
        if _mlir_operation(op).name == "memref.global":
            _mlir_operation(op).erase()

    module_op.verify()


def lower_to_memref_stack_weights(module: Module, fill_value: float) -> None:
    _bufferize_module(module)
    replace_memref_globals_with_stack_alloc_fill(module, fill_value=fill_value)
    _post_cleanup_module(module)


def main():
    parser = argparse.ArgumentParser(description="Compile MNIST model to linalg IR.")
    parser.add_argument(
        "--emit",
        choices=("linalg-tensors", "memref-stack-weights"),
        default="linalg-tensors",
        help=(
            "linalg-tensors: torch-mlir linalg-on-tensors (may print dense_resource<__elided__>). "
            "memref-stack-weights: one-shot-bufferize then replace weight globals with "
            "memref.alloc + linalg.fill (riscv_tests-style placeholders)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MLIR path. Default depends on --emit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size of tracing input tensor.",
    )
    parser.add_argument(
        "--large-elements-limit",
        type=int,
        default=10,
        help="Elide large tensor attributes when printing linalg-on-tensors.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=1.0,
        help="f32/f64 memref fill for --emit memref-stack-weights (placeholder weights).",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            _DEFAULT_MEMREF_OUTPUT
            if args.emit == "memref-stack-weights"
            else _DEFAULT_OUTPUT
        )

    torch.manual_seed(0)
    model = MnistNet().eval()
    example_input = torch.randn(args.batch_size, 1, 28, 28)

    mlir_module = torchscript.compile(
        model,
        example_input,
        output_type="linalg-on-tensors",
        use_tracing=True,
    )

    if args.emit == "memref-stack-weights":
        lower_to_memref_stack_weights(mlir_module, fill_value=args.fill_value)
        asm = mlir_module.operation.get_asm(large_elements_limit=args.large_elements_limit)
    else:
        asm = mlir_module.operation.get_asm(
            large_elements_limit=args.large_elements_limit
        )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(asm)

    print(f"MLIR written to: {args.output}")


if __name__ == "__main__":
    main()
