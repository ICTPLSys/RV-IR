#!/usr/bin/env bash
# PyTorch example -> Linalg(memref) -> ROCC MLIR -> EmitC-style C++ (xDSL).
# Supports MnistNet and tiny ResNet (resnet_simple_to_linalg.py).
#
# Prerequisites:
#   1) RV-IR built: build/bin/torch-mlir-opt (override with --torch-mlir-opt).
#   2) Python with torch + torch_mlir (e.g. conda activate torch-mlir).
#   3) tools/RVV_code_gen_via_MLIR_xDSL: `make install` for .venv used by convert_linalg_mlir_to_c.sh.
#
# Usage:
#   conda activate torch-mlir
#   cd /path/to/RV-IR
#   bash scripts/mnist_torch_linalg_rocc_to_cpp.sh --model mnist
#   bash scripts/mnist_torch_linalg_rocc_to_cpp.sh resnet --wd 16 --type fp
#   bash scripts/mnist_torch_linalg_rocc_to_cpp.sh --model resnet --spatial 5 --pad 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RV_IR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${RV_IR_ROOT}/projects/pt1/examples"
MLIR_OUT_DIR="${EXAMPLES_DIR}/mlir_output"
TOOLS_RVV="${RV_IR_ROOT}/tools/RVV_code_gen_via_MLIR_xDSL"

TORCH_MLIR_OPT="${TORCH_MLIR_OPT:-${RV_IR_ROOT}/build/bin/torch-mlir-opt}"
PYTHON="${PYTHON:-python3}"

MODEL="mnist"
FILL_VALUE="1.0"
WD="16"
TYPE="fp"
VERBOSE=0

# Optional resnet_simple_to_linalg.py overrides (only used when --model resnet)
RESNET_SPATIAL=""
RESNET_PAD=""
RESNET_CONV_CHANNELS=""
RESNET_NUM_CLASSES=""
RESNET_NO_VERIFY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/mnist_torch_linalg_rocc_to_cpp.sh [mnist|resnet] [options]

PyTorch export -> Linalg(memref) -> ROCC MLIR -> C++ under tools/RVV_code_gen_via_MLIR_xDSL/generated/

Model selection (pick one):
  mnist|resnet              Optional first positional argument (same as --model).
  --model mnist|resnet      Default: mnist. "resnet" runs resnet_simple_to_linalg.py.

Common options:
  --torch-mlir-opt PATH     Default: <RV-IR>/build/bin/torch-mlir-opt
  --python PATH             Default: python3
  --fill-value FLOAT        Placeholder weights (default: 1.0); both exporters accept it.
  --wd 8|16|32              convert_linalg_mlir_to_c.sh (default: 16)
  --type int|fp             convert_linalg_mlir_to_c.sh (default: fp)
  -v, --verbose
  -h, --help

ResNet-only (passed to resnet_simple_to_linalg.py when --model resnet):
  --spatial N
  --pad N
  --conv-channels N
  --num-classes N
  --no-verify               Skip MLIR geometry self-check in the Python exporter.

Output layout (under projects/pt1/examples/mlir_output/):
  mnist:  mnist_linalg_memref.mlir  -> mnist_rocc_memref.mlir
  resnet: resnet_simple_memref.mlir -> resnet_simple_rocc_memref.mlir
EOF
  exit 0
}

# Optional positional model for quick invocation: `bash script.sh resnet`
if [[ $# -ge 1 ]] && [[ "$1" =~ ^(mnist|resnet|resnet_simple)$ ]]; then
  case "$1" in
    mnist) MODEL="mnist" ;;
    resnet|resnet_simple) MODEL="resnet" ;;
  esac
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --torch-mlir-opt)
      TORCH_MLIR_OPT="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --fill-value)
      FILL_VALUE="$2"
      shift 2
      ;;
    --wd)
      WD="$2"
      shift 2
      ;;
    --type)
      TYPE="$2"
      shift 2
      ;;
    --spatial)
      RESNET_SPATIAL="$2"
      shift 2
      ;;
    --pad)
      RESNET_PAD="$2"
      shift 2
      ;;
    --conv-channels)
      RESNET_CONV_CHANNELS="$2"
      shift 2
      ;;
    --num-classes)
      RESNET_NUM_CLASSES="$2"
      shift 2
      ;;
    --no-verify)
      RESNET_NO_VERIFY=1
      shift
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

case "$MODEL" in
  mnist) ;;
  resnet|resnet_simple) MODEL="resnet" ;;
  *)
    echo "ERROR: invalid --model or positional: ${MODEL} (use mnist or resnet)" >&2
    exit 1
    ;;
esac

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$EXAMPLES_DIR" ]] || die "Examples dir missing: $EXAMPLES_DIR"
[[ -d "$TOOLS_RVV" ]] || die "RVV tools dir missing: $TOOLS_RVV"
[[ -x "$TORCH_MLIR_OPT" ]] || die "torch-mlir-opt not executable: $TORCH_MLIR_OPT (build RV-IR or set --torch-mlir-opt)"

if [[ "$MODEL" == "mnist" ]]; then
  EXPORT_SCRIPT="mnistnet_to_linalg.py"
  LINALG_MLIR="${MLIR_OUT_DIR}/mnist_linalg_memref.mlir"
  ROCC_MLIR="${MLIR_OUT_DIR}/mnist_rocc_memref.mlir"
  [[ -f "${EXAMPLES_DIR}/mnistnet_to_linalg.py" ]] || die "Missing ${EXAMPLES_DIR}/mnistnet_to_linalg.py"
else
  EXPORT_SCRIPT="resnet_simple_to_linalg.py"
  LINALG_MLIR="${MLIR_OUT_DIR}/resnet_simple_memref.mlir"
  ROCC_MLIR="${MLIR_OUT_DIR}/resnet_simple_rocc_memref.mlir"
  [[ -f "${EXAMPLES_DIR}/resnet_simple_to_linalg.py" ]] || die "Missing ${EXAMPLES_DIR}/resnet_simple_to_linalg.py"
fi

mkdir -p "$MLIR_OUT_DIR"

echo "== Step 1/3: (${MODEL}) ${EXPORT_SCRIPT} -> ${LINALG_MLIR}"
(
  cd "$EXAMPLES_DIR"
  if [[ "$MODEL" == "mnist" ]]; then
    "$PYTHON" "$EXPORT_SCRIPT" \
      --emit memref-stack-weights \
      --fill-value "$FILL_VALUE" \
      --output "$LINALG_MLIR"
  else
    cmd=( "$PYTHON" "$EXPORT_SCRIPT" --emit memref-stack-weights --fill-value "$FILL_VALUE" --output "$LINALG_MLIR" )
    [[ -n "$RESNET_SPATIAL" ]] && cmd+=( --spatial "$RESNET_SPATIAL" )
    [[ -n "$RESNET_PAD" ]] && cmd+=( --pad "$RESNET_PAD" )
    [[ -n "$RESNET_CONV_CHANNELS" ]] && cmd+=( --conv-channels "$RESNET_CONV_CHANNELS" )
    [[ -n "$RESNET_NUM_CLASSES" ]] && cmd+=( --num-classes "$RESNET_NUM_CLASSES" )
    [[ "$RESNET_NO_VERIFY" -eq 1 ]] && cmd+=( --no-verify )
    "${cmd[@]}"
  fi
)

echo "== Step 2/3: Linalg -> ROCC -> ${ROCC_MLIR}"
"$TORCH_MLIR_OPT" "$LINALG_MLIR" --convert-linalg-to-rocc -o "$ROCC_MLIR"

echo "== Step 3/3: ROCC MLIR -> C++ under tools/RVV_code_gen_via_MLIR_xDSL/generated/"
ROCC_ABS="$(cd "$(dirname "$ROCC_MLIR")" && pwd)/$(basename "$ROCC_MLIR")"
(
  cd "$TOOLS_RVV"
  if [[ "$VERBOSE" -eq 1 ]]; then
    ./convert_linalg_mlir_to_c.sh -v "$ROCC_ABS" --wd "$WD" --type "$TYPE"
  else
    ./convert_linalg_mlir_to_c.sh "$ROCC_ABS" --wd "$WD" --type "$TYPE"
  fi
)

OUT_CPP="${TOOLS_RVV}/generated/$(basename "${ROCC_MLIR%.mlir}.cpp")"
echo "Done (model=${MODEL})."
echo "  Linalg MLIR:   $LINALG_MLIR"
echo "  ROCC MLIR:     $ROCC_MLIR"
echo "  Generated C++: $OUT_CPP"
