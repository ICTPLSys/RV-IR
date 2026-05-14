#!/usr/bin/env python3
"""
plot_comparison.py  --  Compare best XDSL-sweep kernel vs OpenBLAS per layer.

Usage:
    python plot_comparison.py \
        --xdsl   <xdsl_sweep.csv>    \
        --oblas  <openblas_out.txt>  \
        --model  <model_name>        \
        --out    <output_prefix>     (optional, defaults to same dir as --xdsl)
        --vlen   256                 (optional)
        --kc     256                 (optional)
        --families  4,4             (optional, MAX_MR,MAX_NR)
        --best-kernel  4x4          (optional, from kc-loop sweep)
        --sweep-log <log.txt>        (optional, C program stdout — contains SWEEP SUMMARY block)

Outputs:
    <out>_perf.pdf   -- bar chart with per-layer best kernel labels
    <out>_perf.txt   -- plain-text summary table + verbatim SWEEP SUMMARY from C program
"""

import argparse
from collections import OrderedDict
import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({'font.size': 14})


# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────

def parse_xdsl_csv(path):
    """
    Parse the XDSL sweep CSV.
    Format: label;m;n;k;gflops  (header '#')
    Returns: dict  (m, n, k) -> best_gflops
    """
    best = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue
            try:
                m, n, k = int(parts[1]), int(parts[2]), int(parts[3])
                gflops = float(parts[4])
            except ValueError:
                continue
            key = (m, n, k)
            if key not in best or gflops > best[key]:
                best[key] = gflops
    return best


def parse_openblas_text(path):
    """
    Parse the OpenBLAS text output.
    Format:  Layer_Name   M   N   K  |  time_us  |  GFLOPS
    Returns: list of dicts {name, m, n, k, gflops} in order
    """
    results = []
    pat = re.compile(
        r'^\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\|[^|]*\|\s*([\d.]+)\s+GFLOPS',
        re.IGNORECASE
    )
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                results.append({
                    'name':   m.group(1),
                    'm':      int(m.group(2)),
                    'n':      int(m.group(3)),
                    'k':      int(m.group(4)),
                    'gflops': float(m.group(5)),
                })
    return results


def extract_sweep_summary(log_path):
    """
    Extract the SWEEP SUMMARY BEST KERNELS block verbatim from the C program log.
    Also returns a dict (m, n, k) -> 'MRxNR' label for per-layer bar annotation.
    """
    if not log_path or not os.path.exists(log_path):
        return '', {}

    lines = open(log_path).readlines()

    # Find the SWEEP SUMMARY block boundaries
    start = None
    end   = None
    for i, line in enumerate(lines):
        # Strip ANSI colour codes before matching
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        if 'SWEEP SUMMARY BEST KERNELS' in clean and start is None:
            # Include the separator line before it
            start = max(0, i - 1)
        if start is not None and i > start + 1:
            # End after the closing separator line
            stripped = clean.strip().strip('-')
            if stripped == '' and i > start + 2:
                end = i + 1
                break

    block_str = ''
    best_uk_map = {}   # (m, n, k) -> 'MR×NR'

    if start is not None:
        block_lines = lines[start : end if end else len(lines)]
        # Strip ANSI codes for the stored text
        clean_block = [re.sub(r'\x1b\[[0-9;]*m', '', l) for l in block_lines]
        block_str = ''.join(clean_block)

        # Parse per-layer lines: | Layer  1 (M=  16, N=  16, K=  15):   1.86e+00 GFLOPS  =>  Kernel [ 4x 4]
        layer_pat = re.compile(
            r'\|\s*Layer\s*(\d+)\s*\(M=\s*(\d+),\s*N=\s*(\d+),\s*K=\s*(\d+)\)'
            r'[^=]*=>\s*Kernel\s*\[\s*(\d+)x\s*(\d+)\]'
        )
        for bl in clean_block:
            lm = layer_pat.search(bl)
            if lm:
                m, n, k = int(lm.group(2)), int(lm.group(3)), int(lm.group(4))
                mr, nr  = int(lm.group(5)), int(lm.group(6))
                best_uk_map[(m, n, k)] = f'{mr}×{nr}'

    return block_str, best_uk_map


# ─────────────────────────────────────────────
# Matching: align xdsl layers to openblas rows
# ─────────────────────────────────────────────

def match_layers(best_gflops, best_uk_map, openblas_rows):
    """
    Align by matching (m, n, k).
    Returns a list of dicts: label, xdsl_gflops, oblas_gflops, m, n, k, best_uk
    """
    rows = []
    for ob in openblas_rows:
        key = (ob['m'], ob['n'], ob['k'])
        rows.append({
            'label':        ob['name'],
            'xdsl_gflops':  best_gflops.get(key, 0.0),
            'oblas_gflops': ob['gflops'],
            'm':   ob['m'],
            'n':   ob['n'],
            'k':   ob['k'],
            'best_uk':  best_uk_map.get(key, ''),
        })
    return rows


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_comparison(rows, model_name, out_prefix, config_str='', best_kernel=''):
    n_layers = len(rows)
    if n_layers == 0:
        print("No layers to plot.")
        return

    labels      = [r['label'] for r in rows]
    xdsl_vals   = [r['xdsl_gflops'] for r in rows]
    oblas_vals  = [r['oblas_gflops'] for r in rows]
    best_uks    = [r['best_uk'] for r in rows]

    x      = np.arange(n_layers)
    width  = 0.35
    fig_w  = max(12, n_layers * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    COLOR_OBLAS = '#888888'   # grey  – OpenBLAS (RVV)
    COLOR_XDSL  = '#6BC5E8'  # cyan  – xDSL micro-kernels

    bars_ob = ax.bar(x - width/2, oblas_vals, width, color=COLOR_OBLAS, label='OpenBLAS (RVV)')
    bars_xl = ax.bar(x + width/2, xdsl_vals,  width, color=COLOR_XDSL,  label='xDSL micro-kernels')

    # Annotate xDSL bars with best kernel label only (no GFLOPS numbers)
    for bar, uk in zip(bars_xl, best_uks):
        if uk:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    uk, ha='center', va='bottom', fontsize=10,
                    color='#1a4f7c', fontweight='bold')

    ax.set_xlabel('Layer')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'{model_name} – xDSL vs OpenBLAS (RVV)', fontsize=12)
    ax.set_xticks(x)

    # Short labels: first letter of model + layer index
    prefix = model_name[0].upper() if model_name else 'L'
    short_labels = [f'{prefix}{i+1}' for i in range(n_layers)]
    ax.set_xticklabels(short_labels, fontsize=12)

    ymax = max(max(xdsl_vals, default=0), max(oblas_vals, default=0))
    ax.set_ylim(0, max(13, ymax * 1.3))
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    pdf_path = out_prefix + '_perf.pdf'
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[plot_comparison] Saved: {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────

SEP = '-' * 90

def write_text_summary(rows, model_name, out_prefix, config_str='', best_kernel='', sweep_block=''):
    txt_path = out_prefix + '_perf.txt'
    with open(txt_path, 'w') as f:
        # Header
        f.write(f"# Performance Comparison: xDSL best kernel vs OpenBLAS\n")
        f.write(f"# Model  : {model_name}\n")
        if config_str:
            f.write(f"# Config : {config_str}\n")
        if best_kernel:
            f.write(f"# Best microkernel (kc-loop sweep): {best_kernel}\n")
        f.write(f"#\n")
        f.write(f"# {'Layer':<25} {'M':>6} {'N':>6} {'K':>6}  {'xDSL_GFLOPS':>12}  {'OpenBLAS_GFLOPS':>15}  {'Ratio':>7}\n")
        f.write(f"# {'-'*25} {'-'*6} {'-'*6} {'-'*6}  {'-'*12}  {'-'*15}  {'-'*7}\n")
        for r in rows:
            ratio = r['xdsl_gflops'] / r['oblas_gflops'] if r['oblas_gflops'] > 0 else float('nan')
            f.write(f"  {r['label']:<25} {r['m']:>6} {r['n']:>6} {r['k']:>6}"
                    f"  {r['xdsl_gflops']:>12.3f}  {r['oblas_gflops']:>15.3f}  {ratio:>7.3f}\n")

        # Embed SWEEP SUMMARY block from C program verbatim
        if sweep_block:
            f.write(f"\n")
            f.write(sweep_block)
        else:
            # Fallback: reconstruct from matched rows
            f.write(f"\n")
            f.write(f" {SEP}\n")
            f.write(f"  SWEEP SUMMARY — BEST xDSL KERNEL PER LAYER\n")
            f.write(f" {SEP}\n")
            for i, r in enumerate(rows):
                uk = r['best_uk'] if r['best_uk'] else 'N/A'
                parts = uk.replace('×', 'x').split('x')
                uk_fmt = f"{parts[0]:>2}x{parts[1]:<2}" if len(parts) == 2 else uk
                f.write(
                    f" | Layer {i+1:>2} (M={r['m']:>4}, N={r['n']:>4}, K={r['k']:>4}):"
                    f"  {r['xdsl_gflops']:>10.2e} GFLOPS  =>  Kernel [{uk_fmt}]\n"
                )
            f.write(f" {SEP}\n")

    print(f"[plot_comparison] Saved: {txt_path}")
    return txt_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Compare xDSL sweep vs OpenBLAS and plot bar chart.')
    ap.add_argument('--xdsl',      required=True, help='Path to xDSL sweep CSV file (layer;m;n;k;gflops)')
    ap.add_argument('--oblas',     required=True, help='Path to OpenBLAS text output file')
    ap.add_argument('--model',     default='',    help='Human-readable model name (e.g. bert_large)')
    ap.add_argument('--out',       default='',    help='Output path prefix (optional)')
    ap.add_argument('--vlen',      default='',    help='Vector length in bits (e.g. 256)')
    ap.add_argument('--kc',        default='',    help='KC profile used (e.g. 256)')
    ap.add_argument('--families',  default='',    help='Kernel families sweep range MAX_MR,MAX_NR (e.g. 4,4)')
    ap.add_argument('--best-kernel', default='', help='Best kernel (MRxNR) from kc-loop sweep, e.g. 4x4')
    ap.add_argument('--sweep-log', default='',   help='Path to C sweep program stdout log (contains SWEEP SUMMARY block)')
    args = ap.parse_args()

    if not os.path.exists(args.xdsl):
        sys.exit(f"ERROR: xDSL file not found: {args.xdsl}")
    if not os.path.exists(args.oblas):
        sys.exit(f"ERROR: OpenBLAS file not found: {args.oblas}")

    model_name = args.model or os.path.splitext(os.path.basename(args.xdsl))[0]

    if args.out:
        out_prefix = args.out
    else:
        base = os.path.splitext(args.xdsl)[0]
        out_prefix = base + '_perf'

    # Build config string
    config_parts = []
    if args.families:
        config_parts.append(f"families={args.families}")
    if args.vlen:
        config_parts.append(f"VLEN={args.vlen}b")
    if args.kc:
        config_parts.append(f"KC={args.kc}")
    config_str = "  |  ".join(config_parts)

    print(f"[plot_comparison] Model     : {model_name}")
    if config_str:
        print(f"[plot_comparison] Config    : {config_str}")
    if args.best_kernel:
        print(f"[plot_comparison] Best kc uK: {args.best_kernel}")
    print(f"[plot_comparison] xDSL CSV  : {args.xdsl}")
    print(f"[plot_comparison] OpenBLAS  : {args.oblas}")

    # Extract SWEEP SUMMARY block + per-layer best kernel from the C program log
    sweep_block, best_uk_from_log = extract_sweep_summary(args.sweep_log)
    if sweep_block:
        print(f"[plot_comparison] Sweep log  : {args.sweep_log}  (SWEEP SUMMARY extracted)")
    elif args.sweep_log:
        print(f"[plot_comparison] [warn] Could not extract SWEEP SUMMARY from: {args.sweep_log}")

    best_gflops = parse_xdsl_csv(args.xdsl)
    openblas_rows = parse_openblas_text(args.oblas)

    if not best_gflops:
        sys.exit("ERROR: No xDSL data parsed – check the CSV format.")
    if not openblas_rows:
        sys.exit("ERROR: No OpenBLAS data parsed – check the output file format.")

    rows = match_layers(best_gflops, best_uk_from_log, openblas_rows)

    write_text_summary(rows, model_name, out_prefix,
                       config_str=config_str, best_kernel=args.best_kernel,
                       sweep_block=sweep_block)
    plot_comparison(rows, model_name, out_prefix,
                    config_str=config_str, best_kernel=args.best_kernel)


if __name__ == '__main__':
    main()
