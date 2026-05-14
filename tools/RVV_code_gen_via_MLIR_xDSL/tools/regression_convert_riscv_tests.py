from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/home/jiashuhui/RVV_code_gen_via_MLIR_xDSL")
TEST_DIR = ROOT / "tests" / "riscv_tests"
OUT_DIR = ROOT / "generated" / "riscv_regression"
CONVERTER_SH = ROOT / "convert_linalg_mlir_to_c.sh"
EXCLUDES = {"llama3_decoder_block.mlir", "llama3_pass_op.mlir"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mlirs = sorted(p for p in TEST_DIR.glob("*.mlir") if p.name not in EXCLUDES)
    if not mlirs:
        print("No MLIR files to test.")
        return 0

    failed: list[tuple[Path, str]] = []
    for mlir in mlirs:
        out_cpp = OUT_DIR / f"{mlir.stem}.cpp"
        cmd = [str(CONVERTER_SH), str(mlir), str(out_cpp)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if p.returncode != 0:
            failed.append((mlir, (p.stderr or p.stdout).strip()))
            print(f"[FAIL] {mlir.name}")
        else:
            print(f"[OK]   {mlir.name}")

    print(f"\nTotal: {len(mlirs)}, Failed: {len(failed)}")
    if failed:
        print("\nFailure details:")
        for mlir, msg in failed:
            print(f"- {mlir.name}: {msg.splitlines()[-1] if msg else 'unknown error'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
