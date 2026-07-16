#!/usr/bin/env python3
"""Export a tiny torch.matmul model and lower it to RAIR IR.

This script demonstrates the shortest useful frontend path:

  PyTorch nn.Module -> torch-mlir linalg-on-tensors -> linalg/memref -> RAIR

Run it from the RV-IR repository root:

  python projects/pt1/examples/matmul_to_rair.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mlir_output"


def _prepend_existing_python_paths() -> None:
    """Prefer the in-tree torch-mlir Python packages when they are built."""
    candidate_paths = [
        REPO_ROOT / "projects/pt1/python",
        REPO_ROOT / "projects/pt1/python/torch_mlir",
        REPO_ROOT / "build/tools/torch-mlir/python_packages/torch_mlir",
    ]
    existing = [str(p) for p in candidate_paths if p.exists()]
    for path in reversed(existing):
        if path not in sys.path:
            sys.path.insert(0, path)
    if existing:
        os.environ["PYTHONPATH"] = os.pathsep.join(existing + sys.path)


_prepend_existing_python_paths()

import torch
import torch.nn as nn
from torch_mlir import torchscript
from torch_mlir.passmanager import PassManager


BUFFERIZE_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries copy-before-write "
    "unknown-type-conversion=identity-layout-map},"
    "canonicalize,cse)"
)


class MatmulModule(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)


def write_module(module, path: Path, *, large_elements_limit: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        module.operation.get_asm(large_elements_limit=large_elements_limit),
        encoding="utf-8",
    )


def bufferize_in_place(module) -> None:
    with module.context:
        PassManager.parse(BUFFERIZE_PIPELINE).run(module.operation)


def run_rair_lowering(torch_mlir_opt: Path, input_mlir: Path, output_mlir: Path) -> None:
    cmd = [
        str(torch_mlir_opt),
        str(input_mlir),
        "--convert-linalg-to-rair",
        "-o",
        str(output_mlir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a small torch.matmul example to RAIR IR."
    )
    parser.add_argument("--m", type=int, default=16, help="Rows of lhs/output.")
    parser.add_argument("--k", type=int, default=32, help="Columns of lhs / rows of rhs.")
    parser.add_argument("--n", type=int, default=64, help="Columns of rhs/output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated MLIR files.",
    )
    parser.add_argument(
        "--torch-mlir-opt",
        type=Path,
        default=REPO_ROOT / "build/bin/torch-mlir-opt",
        help="Path to torch-mlir-opt.",
    )
    parser.add_argument(
        "--large-elements-limit",
        type=int,
        default=32,
        help="Printing limit for dense elements in the linalg-on-tensors dump.",
    )
    args = parser.parse_args()

    if not args.torch_mlir_opt.exists():
        raise FileNotFoundError(
            f"torch-mlir-opt not found: {args.torch_mlir_opt}. "
            "Build it with: ninja -C build torch-mlir-opt"
        )

    torch.manual_seed(0)
    model = MatmulModule().eval()
    lhs = torch.randn(args.m, args.k, dtype=torch.float32)
    rhs = torch.randn(args.k, args.n, dtype=torch.float32)

    module = torchscript.compile(
        model,
        (lhs, rhs),
        output_type="linalg-on-tensors",
        use_tracing=True,
    )

    linalg_mlir = args.output_dir / "matmul_linalg.mlir"
    memref_mlir = args.output_dir / "matmul_linalg_memref.mlir"
    rair_mlir = args.output_dir / "matmul_rair.mlir"

    write_module(module, linalg_mlir, large_elements_limit=args.large_elements_limit)
    bufferize_in_place(module)
    write_module(module, memref_mlir, large_elements_limit=args.large_elements_limit)
    run_rair_lowering(args.torch_mlir_opt, memref_mlir, rair_mlir)

    print("Generated MLIR files:")
    print(f"  linalg-on-tensors: {linalg_mlir}")
    print(f"  linalg/memref:     {memref_mlir}")
    print(f"  RAIR:              {rair_mlir}")
    print()
    print("Inspect RAIR ops with:")
    print(f'  rg "rair\\." {rair_mlir}')


if __name__ == "__main__":
    main()
