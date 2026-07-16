#!/usr/bin/env python3
"""
Gemmini MLIR to C Code Generation Pipeline

Converts MLIR files containing rair.matmul operations to standalone C programs
targeting the Berkeley Gemmini accelerator.

Supports two code generation modes:
  - "isa"  (default): low-level Gemmini ISA calls (mvin/compute/mvout)
  - "auto": high-level tiled_matmul_auto API (simpler, handles tiling internally)

Handles single-matmul and multi-layer (MLP) models automatically.

Usage:
    python gemmini_mlir_to_c.py <input.mlir> [output.c] [-v] [--mode isa|auto]

Pipeline:
    1. Parse MLIR input file (RAIR dialect) via xDSL
    2. Walk the module IR to extract rair.matmul dimensions directly
    3. Generate standalone Gemmini C program via GemminiStrategy
"""

import argparse
import os
import re
import sys
import tempfile

from strategy_code_generator import GemminiStrategy, TensorDescriptor
from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, func, linalg, memref, scf
from xdsl.parser import Parser

from xdsltemplate.dialects.emitc_ext import EmitC_Ext
from xdsltemplate.dialects.riscv import RAIR, BatchMatmulOp, MatmulOp


def _extract_all_matmul_dims(module):
    """Walk the module and extract (M, K, N) from every rair.matmul / rair.batch_matmul."""
    layers = []
    for op in module.walk():
        if isinstance(op, (MatmulOp, BatchMatmulOp)):
            A_type = op.A.type
            B_type = op.B.type
            if not isinstance(A_type, memref.MemRefType):
                continue
            A_shape = A_type.shape.data
            B_shape = B_type.shape.data
            if len(A_shape) == 3:
                M = A_shape[1].data
                K = A_shape[2].data
                N = B_shape[2].data
            else:
                M = A_shape[0].data
                K = A_shape[1].data
                N = B_shape[1].data
            layers.append((M, K, N))
    return layers


def process_mlir_file(
    input_file: str,
    output_file: str,
    mode: str = "isa",
    verbose: bool = False,
) -> None:
    if verbose:
        print("[INFO] Gemmini MLIR to C Code Generation Pipeline")
        print("=" * 70)
        print(f"Input file:     {input_file}")
        print(f"Output file:    {output_file}")
        print(f"Mode:           {mode}")
        print()

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file) as f:
        mlir_content = f.read()

    # Convert custom RAIR syntax to generic MLIR format for xDSL parsing
    if re.search(r"rair\.(batch_matmul|matmul|transpose).*ins\(", mlir_content):
        try:
            import convert_custom_format

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlir", delete=False
            ) as tmp:
                tmp_path = tmp.name
            with open(tmp_path, "w") as tmp:
                tmp.write(mlir_content)
            convert_custom_format.convert_file(tmp_path, tmp_path)
            with open(tmp_path) as f:
                mlir_content = f.read()
            os.unlink(tmp_path)
            if verbose:
                print("[INFO] Converted custom RAIR syntax to generic format")
        except Exception as e:
            if verbose:
                print(f"[WARNING] RAIR custom format conversion failed: {e}")

    if verbose:
        print("[INFO] Step 1: Parsing MLIR input...")

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(emitc.EmitC)
    ctx.load_dialect(RAIR)
    ctx.load_dialect(EmitC_Ext)

    try:
        parser = Parser(ctx, mlir_content)
        module = parser.parse_module()
    except Exception as e:
        print(f"[ERROR] Failed to parse MLIR: {e}")
        sys.exit(1)

    if verbose:
        print("[INFO] MLIR parsed successfully")
        print()

    # Step 2: Extract matmul dimensions from all RAIR ops
    if verbose:
        print("[INFO] Step 2: Extracting matmul dimensions from RAIR ops...")

    layers = _extract_all_matmul_dims(module)
    if not layers:
        print("[ERROR] No rair.matmul or rair.batch_matmul found in input")
        sys.exit(1)

    for idx, (M, K, N) in enumerate(layers):
        if verbose:
            print(f"  Layer {idx}: C[{M}x{N}] = A[{M}x{K}] * B[{K}x{N}]")
    if verbose:
        print()

    # Step 3: Generate Gemmini C code
    if verbose:
        print("[INFO] Step 3: Generating Gemmini C code...")

    strategy = GemminiStrategy()

    if len(layers) == 1:
        M, K, N = layers[0]
        tensors = {
            "A": TensorDescriptor("A", "A", K, M, 1),
            "B": TensorDescriptor("B", "B", N, K, 1),
            "C": TensorDescriptor("C", "C", N, M, 1),
        }
        if mode == "auto":
            c_code = strategy.generate_full_program_auto(tensors)
        else:
            c_code = strategy.generate_full_program(tensors)
    else:
        c_code = strategy.generate_multi_layer_program(layers, mode)

    with open(output_file, "w") as f:
        f.write(c_code)

    if verbose:
        print(f"[INFO] C code written to {output_file}")
        print()
        print("=" * 70)
        print("[SUCCESS] Gemmini C code generation complete!")
        print(f"Output: {output_file}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAIR MLIR to Gemmini baremetal C code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage — low-level ISA mode (single-tile or manual tiling)
  python gemmini_mlir_to_c.py tests/riscv_tests/gemmini_matmul_16x16.mlir

  # High-level tiled_matmul_auto mode (recommended for arbitrary sizes)
  python gemmini_mlir_to_c.py tests/riscv_tests/gemmini_matmul_32x32.mlir --mode auto

  # Multi-layer MLP
  python gemmini_mlir_to_c.py tests/riscv_tests/mlp2_64x832.mlir --mode auto -v

  # Then compile and run on Gemmini:
  #   cp output.c /path/to/gemmini/software/gemmini-rocc-tests/bareMetalC/rair_mlp2.c
  #   cd /path/to/gemmini/software/gemmini-rocc-tests && make bareMetalC
  #   spike --extension=gemmini build/bareMetalC/rair_mlp2-baremetal
        """,
    )

    parser.add_argument(
        "input", help="Input MLIR file containing rair.matmul operations"
    )
    parser.add_argument(
        "output", nargs="?", help="Output C file (default: input with .c extension)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--mode",
        choices=["isa", "auto"],
        default="isa",
        help="Code generation mode: 'isa' for low-level Gemmini ISA, 'auto' for tiled_matmul_auto (default: isa)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".mlir", ".c")
        if args.output == args.input:
            args.output = args.output + ".c"

    process_mlir_file(
        input_file=args.input,
        output_file=args.output,
        mode=args.mode,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
