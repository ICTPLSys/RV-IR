#!/usr/bin/env python3
"""
Generate Linalg / memref MLIR for a *tiny* ResNet-style subgraph (torch-mlir), aligned with
``RVV_code_gen_via_MLIR_xDSL/tests/riscv_tests/resnet_simple_memref.mlir`` by default.

Default geometry (matches RVV test ``resnet_simple_memref.mlir``)
-----------------------------------------------------------------
- Input: **1×3×5×5**
- ``ZeroPad2d(pad)`` on H,W → canvas **(5+2p)×(5+2p)** (default **p=3** → **11×11**)
- Conv **3→C**, k=3, s=1, p=0 → spatial **(5+2p−2) = 9** (with defaults)
- BN + ReLU + global average pool (torch-mlir lowers as ``pooling_nchw_sum`` + div by **9×9**)
- ``Linear(C, num_classes)`` with default **C=4**, **num_classes=4**

Use ``--spatial``, ``--pad``, ``--conv-channels``, ``--num-classes`` to tune; after export the
script checks that the printed MLIR contains the expected rank-4 input / padded canvas / conv
feature map shapes so stale or mismatched outputs fail fast.

Bufferization / placeholder weights follow ``mnistnet_to_linalg.py`` (reuse
``lower_to_memref_stack_weights``).
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MLIR_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "mlir_output")
_DEFAULT_LINALG = os.path.join(_DEFAULT_MLIR_OUTPUT_DIR, "resnet_simple_linalg.mlir")
_DEFAULT_MEMREF = os.path.join(_DEFAULT_MLIR_OUTPUT_DIR, "resnet_simple_memref.mlir")

home_dir = os.path.expanduser("~")
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
from torch_mlir import torchscript

# Reuse bufferize + memref.global → alloc/fill from mnist example.
sys.path.insert(0, _SCRIPT_DIR)
from mnistnet_to_linalg import lower_to_memref_stack_weights


def _padded_spatial(spatial: int, pad: int) -> int:
    return spatial + 2 * pad


def _conv_spatial(spatial: int, pad: int, kernel: int = 3) -> int:
    """Conv k=3 s=1, valid on padded canvas."""
    return _padded_spatial(spatial, pad) - kernel + 1


def _verify_exported_mlir(
    asm: str,
    *,
    emit: str,
    spatial: int,
    pad: int,
    conv_channels: int,
    num_classes: int,
    kernel: int = 3,
) -> None:
    padded = _padded_spatial(spatial, pad)
    conv_sp = _conv_spatial(spatial, pad, kernel)
    kind = "tensor" if emit == "linalg-tensors" else "memref"

    checks: list[tuple[str, str]] = [
        (f"func.func @forward(%arg0: {kind}<1x3x{spatial}x{spatial}", "forward input signature"),
        (f"{kind}<1x3x{padded}x{padded}x", f"padded canvas ({kind})"),
        (f"{kind}<1x{conv_channels}x{conv_sp}x{conv_sp}x", f"post-conv activations ({kind})"),
        (f") -> {kind}<1x{num_classes}xf32>", "return logits type"),
    ]
    missing = [label for needle, label in checks if needle not in asm]
    if missing:
        raise RuntimeError(
            "Exported MLIR failed geometry self-check (sizes do not match requested config). "
            f"Missing patterns for: {missing}. "
            f"Expected spatial={spatial}, pad={pad} → padded={padded}, conv_sp={conv_sp}, "
            f"C={conv_channels}, classes={num_classes}. "
            "If you changed hyperparameters, disable checks with --no-verify."
        )


class ResNetSimple(nn.Module):
    """Minimal ResNet-style forward; dimensions are explicit ctor args."""

    def __init__(
        self,
        *,
        spatial_pad: int,
        conv_out_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        conv_stride: int = 1,
    ) -> None:
        super().__init__()
        p = spatial_pad
        self.pad = nn.ZeroPad2d((p, p, p, p))
        self.conv = nn.Conv2d(
            3,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=0,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(conv_out_channels, affine=True)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torch-mlir export of a tiny ResNet-style block (resnet_simple_memref-like)."
    )
    parser.add_argument(
        "--emit",
        choices=("linalg-tensors", "memref-stack-weights"),
        default="linalg-tensors",
        help="Same semantics as mnistnet_to_linalg.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MLIR path (default depends on --emit).",
    )
    parser.add_argument(
        "--spatial",
        type=int,
        default=5,
        help="Square input H=W (RVV resnet_simple_memref.mlir uses 5).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=3,
        help="ZeroPad2d per side on H and W (RVV test uses 3 → 11×11 canvas for 5×5 in).",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        default=4,
        help="Conv output / BN channels (RVV test uses 4).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="FC output size (RVV test uses 4 logits).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-export shape checks on printed MLIR.",
    )
    parser.add_argument(
        "--large-elements-limit",
        type=int,
        default=10,
        help="Elide large dense attributes when printing.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=1.0,
        help="Placeholder fill for memref-stack-weights mode.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = _DEFAULT_MEMREF if args.emit == "memref-stack-weights" else _DEFAULT_LINALG

    spatial = args.spatial
    pad = args.pad
    padded = _padded_spatial(spatial, pad)
    conv_sp = _conv_spatial(spatial, pad)

    torch.manual_seed(0)
    model = ResNetSimple(
        spatial_pad=pad,
        conv_out_channels=args.conv_channels,
        num_classes=args.num_classes,
    ).eval()

    example = torch.randn(1, 3, spatial, spatial)
    print(
        "Tracing shape:",
        tuple(example.shape),
        f"| pad={pad} → canvas {padded}×{padded}, conv out {conv_sp}×{conv_sp}, "
        f"C={args.conv_channels}, classes={args.num_classes}",
    )

    mlir_module = torchscript.compile(
        model,
        example,
        output_type="linalg-on-tensors",
        use_tracing=True,
    )

    if args.emit == "memref-stack-weights":
        lower_to_memref_stack_weights(mlir_module, fill_value=args.fill_value)

    asm = mlir_module.operation.get_asm(large_elements_limit=args.large_elements_limit)

    if not args.no_verify:
        _verify_exported_mlir(
            asm,
            emit=args.emit,
            spatial=spatial,
            pad=pad,
            conv_channels=args.conv_channels,
            num_classes=args.num_classes,
        )

    div_area = float(conv_sp * conv_sp)
    banner = (
        "// Auto-generated by resnet_simple_to_linalg.py (torch-mlir). "
        f"emit={args.emit}. Not bit-identical to tests/riscv_tests/resnet_simple.mlir.\n"
        "// Hand-tuned (defaults = RVV tests/riscv_tests/resnet_simple_memref.mlir): "
        f"input 1x3x{spatial}x{spatial}, padded {padded}x{padded}, conv {conv_sp}x{conv_sp}, "
        f"pool {conv_sp}x{conv_sp} / {div_area:g}; logits 1x{args.num_classes} "
        f"(FC {args.conv_channels}x{args.num_classes} + bias {args.num_classes}).\n"
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(banner)
        f.write(asm)

    print(f"MLIR written to: {args.output}")


if __name__ == "__main__":
    main()
