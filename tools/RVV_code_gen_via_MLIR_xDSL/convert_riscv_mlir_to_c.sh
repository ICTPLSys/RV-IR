#!/bin/bash

###############################################################################
# RISCV MLIR to C Conversion Script
#
# This script converts MLIR files containing rocc.batch_matmul operations
# to C code using the xDSL framework and mlir-translate.
#
# Usage:
#   ./convert_riscv_mlir_to_c.sh <input.mlir> [output.cpp]
#
# Example:
#   ./convert_riscv_mlir_to_c.sh tests/riscv_batch_matmul_test.mlir
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERBOSE=0
MLIR_TRANSLATE="mlir-translate"
STRATEGY="simple"

# Parse arguments
INPUT_FILE=""
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <input.mlir> [output.cpp]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose          Enable verbose output"
            echo "  -s, --strategy STRATEGY  GEMM execution strategy (simple, workload, blocked)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Arguments:"
            echo "  input.mlir             Input MLIR file containing rocc.batch_matmul"
            echo "  output.cpp             Output C++ file (default: input.cpp)"
            echo ""
            echo "Available strategies:"
            echo "  simple   - Direct computation (default, for small matrices)"
            echo "  workload - With data movement and iteration (for benchmarking)"
            echo "  blocked  - Tiled computation (for large matrices)"
            echo ""
            echo "Examples:"
            echo "  $0 tests/riscv_batch_matmul_test.mlir"
            echo "  $0 tests/riscv_batch_matmul_test.mlir --strategy blocked"
            echo "  $0 tests/riscv_batch_matmul_test.mlir -s workload -v"
            exit 0
            ;;
        -s|--strategy)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo -e "${RED}[ERROR]${NC} --strategy requires an argument"
                exit 1
            fi
            STRATEGY="$2"
            shift 2
            ;;
        -*)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            elif [ -z "$OUTPUT_FILE" ]; then
                OUTPUT_FILE="$1"
            else
                echo -e "${RED}[ERROR]${NC} Too many arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check input file
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} No input file specified"
    echo "Usage: $0 <input.mlir> [output.cpp]"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Input file not found: $INPUT_FILE"
    exit 1
fi

# Set output file if not specified
# if [ -z "$OUTPUT_FILE" ]; then
#     OUTPUT_FILE="${INPUT_FILE%.mlir}.cpp"
#     if [ "$OUTPUT_FILE" = "$INPUT_FILE" ]; then
#         OUTPUT_FILE="${INPUT_FILE}.cpp"
#     fi
# fi
if [ -z "$OUTPUT_FILE" ]; then
    mkdir -p generated
    filename=$(basename -- "$INPUT_FILE")
    filename_noext="${filename%.mlir}"
    OUTPUT_FILE="generated/${filename_noext}.cpp"
fi

# Print header
if [ $VERBOSE -eq 1 ]; then
    echo -e "${BLUE}[INFO]${NC} RoCC MLIR to C Code Generation Pipeline"
    echo "=================================================="
    echo -e "Input file:     ${INPUT_FILE}"
    echo -e "Output file:    ${OUTPUT_FILE}"
    echo -e "Strategy:       ${STRATEGY}"
    echo -e "Working dir:    ${PWD}"
    echo ""
fi

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/riscv_mlir_to_c.py"
CONVERT_FORMAT_SCRIPT="$SCRIPT_DIR/convert_custom_format.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}[ERROR]${NC} Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

INPUT_FILE_FOR_PROCESSING="$INPUT_FILE"
# Check if input file uses custom RoCC dialect format
if grep -q -E "rocc\.(batch_matmul|transpose).*ins\(" "$INPUT_FILE" 2>/dev/null; then
    if [ -f "$CONVERT_FORMAT_SCRIPT" ]; then
        if [ $VERBOSE -eq 1 ]; then
            echo -e "${BLUE}[INFO]${NC} Converting custom RoCC format to generic format..."
        fi
        TEMP_CONVERTED="${TMPDIR:-/tmp}/riscv_mlir_converted_$$$.mlir"
        # Run Python directly without activating venv here
        # The venv will be activated later for the main conversion
        PYTHON_BIN="python3"
        if [ -d "$SCRIPT_DIR/.venv" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
        fi
        $PYTHON_BIN "$CONVERT_FORMAT_SCRIPT" "$INPUT_FILE" "$TEMP_CONVERTED"
        if [ $? -eq 0 ]; then
            INPUT_FILE_FOR_PROCESSING="$TEMP_CONVERTED"
            if [ $VERBOSE -eq 1 ]; then
                echo -e "${BLUE}[INFO]${NC} Format conversion complete"
            fi
        else
            echo -e "${YELLOW}[WARNING]${NC} Format conversion failed, trying original file"
            INPUT_FILE_FOR_PROCESSING="$INPUT_FILE"
        fi
    fi
fi

# Activate virtual environment if it exists
VENV_DIR="$SCRIPT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    if [ $VERBOSE -eq 1 ]; then
        echo -e "${BLUE}[INFO]${NC} Activating virtual environment: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
fi

# Check if xDSL is available
if ! python3 -c "import xdsl" 2>/dev/null; then
    echo -e "${RED}[ERROR]${NC} xDSL package not found. Please install it first."
    echo -e "${YELLOW}[INFO]${NC} Run: pip install xdsl"
    exit 1
fi

# Run Python conversion script
VERBOSE_FLAG=""
STRATEGY_FLAG="--strategy $STRATEGY"
if [ $VERBOSE -eq 1 ]; then
    VERBOSE_FLAG="--verbose"
fi

python3 "$PYTHON_SCRIPT" "$INPUT_FILE_FOR_PROCESSING" "$OUTPUT_FILE" $STRATEGY_FLAG $VERBOSE_FLAG

# Check if conversion was successful
if [ $? -eq 0 ]; then
    if [ $VERBOSE -eq 1 ]; then
        echo ""
        echo -e "${GREEN}[SUCCESS]${NC} Conversion completed successfully!"
        echo -e "Output file: ${OUTPUT_FILE}"
    else
        echo -e "${GREEN}[SUCCESS]${NC} Generated: ${OUTPUT_FILE}"
    fi

    # Print file size info
    if command -v wc &> /dev/null; then
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo -e "Lines: ${LINE_COUNT}, Size: ${FILE_SIZE}"
    fi
else
    echo -e "${RED}[ERROR]${NC} Conversion failed"
    exit 1
fi
