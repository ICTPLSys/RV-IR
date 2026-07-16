#!/usr/bin/env python3
"""
RAIR MLIR to C Code Generation Pipeline

This script converts MLIR files containing rair.batch_matmul operations to C code
using the xDSL framework and mlir-translate.

Usage:
    python riscv_mlir_to_c.py <input.mlir> [output.cpp]

Pipeline:
    1. Parse MLIR input file
    2. Apply RAIR -> EmitC transformation passes
    3. Dump transformed MLIR
    4. Generate C code manually from transformed MLIR
"""

import argparse
import os
import re
import sys
import tempfile

# Import unified code generator (contains both emitc and strategy-based generation)
from strategy_code_generator import StrategyRegistry, generate_c_with_strategy
from xdsl.context import Context
from xdsl.dialects import arith, builtin, emitc, func, linalg, memref, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.printer import Printer

from xdsltemplate.dialects.emitc_ext import EmitC_Ext

# Import RAIR dialect and transforms
from xdsltemplate.dialects.riscv import RAIR
from xdsltemplate.transforms.arith_to_emitc import ArithToEmitCPass
from xdsltemplate.transforms.memref_load_to_emitc import MemrefLoadToEmitcPass
from xdsltemplate.transforms.memref_store_to_emitc import MemrefStoreToEmitcPass
from xdsltemplate.transforms.memref_to_emitc import (
    ConvertMemRefTypeToEmitCPtr,
    MemRefToEmitCPass,
    RemoveUnrealizedConversionCasts,
)
from xdsltemplate.transforms.riscv_to_emitc import RAIRToEmitCPass
from xdsltemplate.transforms.scf_to_emitc import SCFToEmitCPass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MLIR_TRANSLATE = "mlir-translate"  # Assumes it's in PATH


# ============================================================================
# Custom Pass for Function Signature Conversion
# ============================================================================


class ConvertMemRefFuncSignatures(ModulePass):
    """Convert memref types to emitc.ptr in function signatures (after rair-to-emitc)"""

    name = "convert-memref-func-sigs"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertMemRefTypeToEmitCPtr(), apply_recursively=True
        ).rewrite_module(op)


class RemoveUnrealizedCasts(ModulePass):
    """Pass to remove unrealized conversion casts after function signature conversion"""

    name = "remove-unrealized-casts"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            RemoveUnrealizedConversionCasts(), apply_recursively=True
        ).rewrite_module(op)


# ============================================================================
# MLIR Processing Pipeline
# ============================================================================


def process_mlir_file(
    input_file: str,
    output_file: str,
    mlir_translate_path: str,
    strategy: str = "simple",
    verbose: bool = False,
) -> None:
    """
    Process an MLIR file containing rair.batch_matmul operations

    Args:
        input_file: Path to input MLIR file
        output_file: Path to output C++ file
        mlir_translate_path: Path to mlir-translate executable
        strategy: GEMM execution strategy (simple, workload, blocked)
        verbose: Enable verbose output
    """
    # ================================================================
    # Step 1: Parse MLIR input
    # ================================================================
    if verbose:
        print("[INFO] RAIR MLIR to C Code Generation Pipeline")
        print("=" * 70)
        print(f"Input file:     {input_file}")
        print(f"Output file:    {output_file}")
        print(f"Working dir:    {os.getcwd()}")
        print()

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    # Read MLIR file
    with open(input_file) as f:
        mlir_content = f.read()

    if re.search(r"rair\.(batch_matmul|transpose).*ins\(", mlir_content):
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
        except Exception as e:
            print(f"[WARNING] RAIR custom format conversion failed: {e}")

    if verbose:
        print("[INFO] Step 1: Parsing MLIR input...")

    # Create xDSL context and load dialects
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(emitc.EmitC)
    ctx.load_dialect(RAIR)
    ctx.load_dialect(EmitC_Ext)  # Load emitc_ext for emitc.constant operation

    # Parse MLIR
    try:
        parser = Parser(ctx, mlir_content)
        module = parser.parse_module()
    except Exception as e:
        print(f"[ERROR] Failed to parse MLIR: {e}")
        sys.exit(1)

    if verbose:
        print("[INFO] ✓ MLIR parsed successfully")
        print()

    # ================================================================
    # Step 2: Apply transformation passes
    # ================================================================
    if verbose:
        print("[INFO] Step 2: Applying transformation passes...")

    pass_pipeline = [
        ("memref-to-emitc-casts", MemRefToEmitCPass()),  # Create pointer casts
        ("rair-to-emitc", RAIRToEmitCPass()),
        (
            "memref-to-emitc-funcs",
            ConvertMemRefFuncSignatures(),
        ),  # Convert function signatures
        ("remove-unrealized-casts", RemoveUnrealizedCasts()),  # Remove no-op casts
        ("arith-to-emitc", ArithToEmitCPass()),
        ("memref-load-to-emitc", MemrefLoadToEmitcPass()),
        ("memref-store-to-emitc", MemrefStoreToEmitcPass()),
        ("scf-to-emitc", SCFToEmitCPass()),
    ]

    for i, (name, pass_) in enumerate(pass_pipeline):
        try:
            pass_.apply(ctx, module)
            if verbose:
                print(f"  [{i + 1}/{len(pass_pipeline)}] ✓ {name}")
        except Exception as e:
            print(f"  [{i + 1}/{len(pass_pipeline)}] ✗ {name} - {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            print(f"[WARNING] Pass {name} failed, continuing...")

    if verbose:
        print("[INFO] ✓ All passes applied")
        print()

    # ================================================================
    # Step 3: Dump transformed MLIR to temporary file
    # ================================================================
    if verbose:
        print("[INFO] Step 3: Dumping transformed MLIR...")

    temp_mlir = output_file.replace(".cpp", ".mlir").replace(".c", ".mlir")

    try:
        with open(temp_mlir, "w") as f:
            Printer(stream=f, print_generic_format=True).print_op(module)
        if verbose:
            print(f"[INFO] ✓ Transformed MLIR written to {temp_mlir}")
    except Exception as e:
        print(f"[ERROR] Failed to write MLIR: {e}")
        sys.exit(1)

    # ================================================================
    # Step 4: Generate C code from transformed MLIR
    # ================================================================
    if verbose:
        print(
            f"[INFO] Step 4: Generating C code from transformed MLIR (strategy: {strategy})..."
        )

    try:
        with open(temp_mlir) as f:
            mlir_content = f.read()

        # Use strategy-based code generator
        c_code = generate_c_with_strategy(mlir_content, strategy, verbose)

        # Write generated C code
        with open(output_file, "w") as f:
            f.write(c_code)

        if verbose:
            print(f"[INFO] ✓ C code written to {output_file}")

    except Exception as e:
        print(f"[ERROR] C code generation failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    if verbose:
        print()
        print("=" * 70)
        print("[SUCCESS] Code generation complete!")
        print(f"Output: {output_file}")
        print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAIR MLIR to C code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python riscv_mlir_to_c.py input.mlir output.cpp

  # With verbose output
  python riscv_mlir_to_c.py input.mlir output.cpp --verbose

  # Specify custom mlir-translate path
  python riscv_mlir_to_c.py input.mlir output.cpp \\
      --mlir-translate /path/to/mlir-translate
        """,
    )

    parser.add_argument(
        "input", help="Input MLIR file containing rair.batch_matmul operations"
    )
    parser.add_argument(
        "output", nargs="?", help="Output C++ file (default: input.cpp)"
    )
    parser.add_argument(
        "--mlir-translate",
        default=DEFAULT_MLIR_TRANSLATE,
        help=f"Path to mlir-translate executable (default: {DEFAULT_MLIR_TRANSLATE})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--strategy",
        choices=StrategyRegistry.available_strategies(),
        default="simple",
        help="GEMM execution strategy (default: simple)",
    )

    args = parser.parse_args()

    # Determine output file
    if args.output is None:
        args.output = args.input.replace(".mlir", ".cpp")
        if args.output == args.input:
            args.output = args.output + ".cpp"

    # Process the file
    process_mlir_file(
        input_file=args.input,
        output_file=args.output,
        mlir_translate_path=args.mlir_translate,
        strategy=args.strategy,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
