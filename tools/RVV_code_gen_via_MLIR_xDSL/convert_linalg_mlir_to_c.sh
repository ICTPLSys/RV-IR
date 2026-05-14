#!/bin/bash

###############################################################################
# Linalg Generic MLIR to C Conversion Script
#
# This script converts MLIR files containing linalg.generic operations
# to C code using the xDSL framework.
#
# Usage:
#   ./convert_linalg_mlir_to_c.sh <input.mlir> [output.cpp]
#
# Example:
#   ./convert_linalg_mlir_to_c.sh tests/test_linalg_elementwise_add.mlir
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
WD=8
TENSOR_TYPE="int"

# Parse arguments
INPUT_FILE=""
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --wd)
            if [ -z "$2" ]; then
                echo -e "${RED}[ERROR]${NC} --wd requires a value (8/16/32)"
                exit 1
            fi
            WD="$2"
            shift 2
            ;;
        --type)
            if [ -z "$2" ]; then
                echo -e "${RED}[ERROR]${NC} --type requires a value (int/fp)"
                exit 1
            fi
            TENSOR_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <input.mlir> [output.cpp]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose          Enable verbose output"
            echo "  --wd <8|16|32>         Set generated tensor width (default: 8)"
            echo "  --type <int|fp>        Set generated tensor type (default: int)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Arguments:"
            echo "  input.mlir             Input MLIR file containing linalg.generic operations"
            echo "  output.cpp             Output C++ file (default: generated/input.cpp)"
            echo ""
            echo "Examples:"
            echo "  $0 tests/test_linalg_elementwise_add.mlir"
            echo "  $0 tests/test_linalg_square.mlir -v"
            echo "  $0 tests/riscv_tests/mnist_linalg.mlir --wd 16"
            echo "  $0 tests/riscv_tests/mnist_linalg.mlir --wd 16 --type fp"
            echo "  $0 tests/test_linalg_reduce_sum.mlir output/reduce_sum.cpp"
            exit 0
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

# Validate wd
if [[ "$WD" != "8" && "$WD" != "16" && "$WD" != "32" ]]; then
    echo -e "${RED}[ERROR]${NC} Invalid --wd value: $WD (expected 8, 16, or 32)"
    exit 1
fi

# Validate type
if [[ "$TENSOR_TYPE" != "int" && "$TENSOR_TYPE" != "fp" ]]; then
    echo -e "${RED}[ERROR]${NC} Invalid --type value: $TENSOR_TYPE (expected int or fp)"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Input file not found: $INPUT_FILE"
    exit 1
fi

# Set output file if not specified
if [ -z "$OUTPUT_FILE" ]; then
    mkdir -p generated
    filename=$(basename -- "$INPUT_FILE")
    filename_noext="${filename%.mlir}"
    OUTPUT_FILE="generated/${filename_noext}.cpp"
fi

# Print header
if [ $VERBOSE -eq 1 ]; then
    echo -e "${BLUE}[INFO]${NC} Linalg Generic MLIR to C Code Generation Pipeline"
    echo "=================================================="
    echo -e "Input file:     ${INPUT_FILE}"
    echo -e "Output file:    ${OUTPUT_FILE}"
    echo -e "Tensor width:   ${WD}"
    echo -e "Tensor type:    ${TENSOR_TYPE}"
    echo -e "Working dir:    ${PWD}"
    echo ""
fi

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/linalg_mlir_to_c.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}[ERROR]${NC} Python script not found: $PYTHON_SCRIPT"
    exit 1
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
if [ $VERBOSE -eq 1 ]; then
    VERBOSE_FLAG="--verbose"
fi

python3 "$PYTHON_SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE" $VERBOSE_FLAG --wd "$WD" --type "$TENSOR_TYPE"

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
