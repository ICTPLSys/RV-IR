#!/usr/bin/env python3
"""
Convert RAIR dialect custom assembly to generic MLIR format for xDSL parsing.

This script preprocesses MLIR files to convert custom RAIR dialect syntax
to generic MLIR format that xDSL can parse.
"""

import re
import sys


def convert_rair_batch_matmul(line):
    """
    Convert custom format:
        rair.batch_matmul ins(%A, %B : typeA, typeB) outs(%C : typeC)
    To generic format:
        "rair.batch_matmul"(%A, %B, %C) : (typeA, typeB, typeC) -> typeC
    """
    pattern = (
        r"(?:%(\w+)\s*=\s*)?rair\.batch_matmul\s+ins\(([^)]+)\)\s+outs\(([^)]+)\)"
    )

    def replace_func(match):
        result_var = match.group(1)
        ins_part = match.group(2)
        outs_part = match.group(3)

        ins_match = re.match(
            r"\s*([^,]+),\s*([^,]+)\s*:\s*([^,]+),\s*(.+)\s*", ins_part
        )
        if not ins_match:
            return match.group(0)

        op_a = ins_match.group(1).strip()
        op_b = ins_match.group(2).strip()
        type_a = ins_match.group(3).strip()
        type_b = ins_match.group(4).strip()

        outs_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", outs_part)
        if not outs_match:
            return match.group(0)

        op_c = outs_match.group(1).strip()
        type_c = outs_match.group(2).strip()

        if result_var:
            return f'%{result_var} = "rair.batch_matmul"({op_a}, {op_b}, {op_c}) : ({type_a}, {type_b}, {type_c}) -> {type_c}'
        return f'"rair.batch_matmul"({op_a}, {op_b}, {op_c}) : ({type_a}, {type_b}, {type_c}) -> ()'

    return re.sub(pattern, replace_func, line)


def convert_rair_matmul(line):
    """
    Convert custom format:
        rair.matmul ins(%A, %B : typeA, typeB) outs(%C : typeC)
    To generic format:
        %result = "rair.matmul"(%A, %B, %C) : (typeA, typeB, typeC) -> typeC
    """
    pattern = r"(?:%(\w+)\s*=\s*)?rair\.matmul\s+ins\(([^)]+)\)\s+outs\(([^)]+)\)"

    def replace_func(match):
        result_var = match.group(1)
        ins_part = match.group(2)
        outs_part = match.group(3)

        ins_match = re.match(
            r"\s*([^,]+),\s*([^,]+)\s*:\s*([^,]+),\s*(.+)\s*", ins_part
        )
        if not ins_match:
            return match.group(0)

        op_a = ins_match.group(1).strip()
        op_b = ins_match.group(2).strip()
        type_a = ins_match.group(3).strip()
        type_b = ins_match.group(4).strip()

        outs_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", outs_part)
        if not outs_match:
            return match.group(0)

        op_c = outs_match.group(1).strip()
        type_c = outs_match.group(2).strip()

        if result_var:
            return f'%{result_var} = "rair.matmul"({op_a}, {op_b}, {op_c}) : ({type_a}, {type_b}, {type_c}) -> {type_c}'
        return f'"rair.matmul"({op_a}, {op_b}, {op_c}) : ({type_a}, {type_b}, {type_c}) -> ()'

    return re.sub(pattern, replace_func, line)


def convert_rair_transpose(line):
    """
    Convert custom format:
        rair.transpose ins(%input : typeInput) outs(%output : typeOutput) {permutation = array<i64: ...>}
        OR
        rair.transpose ins(%input : typeInput) outs(%output : typeOutput) permutation = [...]
    To generic format:
        "rair.transpose"(%input, %output) {permutation = array<i64: ...>} : (typeInput, typeOutput) -> ()
    """
    if "permutation" not in line:
        return line

    parts = line.split("permutation", 1)
    if len(parts) != 2:
        return line

    main_part = parts[0].strip()
    perm_part = parts[1].strip()

    pattern = r"rair\.transpose\s+ins\(([^)]+)\)\s+outs\(([^)]+)\)"

    match = re.match(pattern, main_part)
    if not match:
        return line

    op_input_raw = match.group(1)
    op_output_raw = match.group(2)

    ins_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", op_input_raw)
    if not ins_match:
        return line
    op_input = ins_match.group(1).strip()
    type_input = ins_match.group(2).strip()

    outs_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", op_output_raw)
    if not outs_match:
        return line
    op_output = outs_match.group(1).strip()
    type_output = outs_match.group(2).strip()

    perm_match = re.search(r"=\s*(\[[^\]]+\]|array<[^>]+>)", perm_part)
    if perm_match:
        perm_value = perm_match.group(1)
        if perm_value.startswith("["):
            values = perm_value.strip("[]").strip()
            perm_value = f"array<i64: {values}>"
        attrs_generic = f"{{permutation = {perm_value}}}"
    else:
        attrs_generic = ""

    return f'"rair.transpose"({op_input}, {op_output}) {attrs_generic} : ({type_input}, {type_output}) -> ()'


def convert_linalg_fill(line):
    """Convert linalg.fill custom assembly to generic form."""
    pattern = r"linalg\.fill\s+ins\(([^)]+)\)\s+outs\(([^)]+)\)"

    def replace_func(match):
        ins_part = match.group(1)
        outs_part = match.group(2)

        ins_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", ins_part)
        if not ins_match:
            return match.group(0)

        value = ins_match.group(1).strip()
        value_type = ins_match.group(2).strip()

        outs_match = re.match(r"\s*([^:]+)\s*:\s*(.+)\s*", outs_part)
        if not outs_match:
            return match.group(0)

        memref = outs_match.group(1).strip()
        memref_type = outs_match.group(2).strip()

        return f'"linalg.fill"({value}, {memref}) <{{operandSegmentSizes = array<i32: 1, 1>}}> : ({value_type}, {memref_type}) -> ()'

    return re.sub(pattern, replace_func, line)


def convert_file(input_file, output_file):
    """Convert an entire MLIR file from custom to generic format"""
    with open(input_file) as f:
        content = f.read()

    content = re.sub(r"\bemitc\.constant\b", "emitc_ext.constant", content)

    def combine_transpose_lines(match):
        full_match = match.group(0)
        return " ".join(full_match.split())

    content = re.sub(
        r"rair\.transpose\s+ins\([^)]+\)[^{]*\{[^\}]*\}",
        combine_transpose_lines,
        content,
        flags=re.DOTALL,
    )

    lines = content.splitlines(keepends=True)
    converted_lines = []
    for line in lines:
        line = convert_rair_batch_matmul(line)
        line = convert_rair_matmul(line)
        line = convert_rair_transpose(line)
        line = convert_linalg_fill(line)
        converted_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(converted_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: convert_custom_format.py <input.mlir> [output.mlir]")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        if input_file.endswith(".mlir"):
            output_file = input_file[:-5] + "_generic.mlir"
        else:
            output_file = input_file + "_generic.mlir"

    try:
        convert_file(input_file, output_file)
        print(f"Converted {input_file} -> {output_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
