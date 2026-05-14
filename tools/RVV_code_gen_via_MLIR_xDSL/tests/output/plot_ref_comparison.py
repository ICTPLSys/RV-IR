#!/usr/bin/env python3
"""
plot_ref_comparison.py  --  xDSL vs Reference micro-kernel performance (selected kernels).

Reads the uK_*_heatmap source file produced by test_gemm (b0 section):
  <MRxNR>_b0   <t1> µs   <t2> µs   <xDSL_GFLOPS> GFLOP/s   <ref_GFLOPS> GFLOP/s

Produces a grouped bar chart:
  - Blue  bars : xDSL micro-kernels
  - Orange bars: Reference micro-kernels
  - Black line  : Speedup ratio (xDSL / Ref), right Y-axis

Usage:
    python plot_ref_comparison.py <uK_file.txt>
                                  [--kernels 8x8,8x15,16x8,16x16]
                                  [--out <prefix>]

If --kernels is omitted, the script auto-selects up to 6 representative kernels
(overall best, best per row, best per column, and largest dimension).
"""

import argparse
import os
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 13})

# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────

def parse_uk_file(path):
    """
    Parse b0 rows from the uK test_gemm output.
    Line format:
        1x1_b0    1.00 µs    1.04 µs    0.512 GFLOP/s    0.491 GFLOP/s
    Returns: dict  (mr, nr) -> {'xdsl': float, 'ref': float}
    """
    data = {}
    pat = re.compile(
        r'^(\d+)x(\d+)_b0'          # MR x NR
        r'.*?'
        r'([\d.]+)\s+GFLOP/s'       # first  GFLOP/s  (xDSL)
        r'.*?'
        r'([\d.]+)\s+GFLOP/s'       # second GFLOP/s  (ref)
    )
    in_b0 = False
    with open(path) as f:
        for line in f:
            # Strip ANSI codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            if '--- Profiling b0 kernels ---' in line:
                in_b0 = True
                continue
            if in_b0 and '--- Profiling' in line and 'b0' not in line:
                break  # left b0 section
            if not in_b0:
                continue
            m = pat.match(line)
            if m:
                mr, nr = int(m.group(1)), int(m.group(2))
                xdsl = float(m.group(3))
                ref  = float(m.group(4))
                data[(mr, nr)] = {'xdsl': xdsl, 'ref': ref}
    return data


def auto_select_kernels(data, n=6):
    """
    Auto-select up to `n` representative kernels:
      - Overall best xDSL
      - Best per unique MR
      - Best per unique NR
      - Then fill remaining with next-highest xDSL performers
    Returns a sorted list of (mr, nr) tuples.
    """
    if not data:
        return []

    scored = sorted(data.items(), key=lambda kv: kv[1]['xdsl'], reverse=True)

    selected = set()
    mrs_seen = set()
    nrs_seen = set()

    # Always include overall best
    selected.add(scored[0][0])
    mrs_seen.add(scored[0][0][0])
    nrs_seen.add(scored[0][0][1])

    # Best per MR
    for (mr, nr), _ in scored:
        if mr not in mrs_seen and len(selected) < n:
            selected.add((mr, nr))
            mrs_seen.add(mr)

    # Best per NR
    for (mr, nr), _ in scored:
        if nr not in nrs_seen and len(selected) < n:
            selected.add((mr, nr))
            nrs_seen.add(nr)

    # Fill remaining with next-best
    for (mr, nr), _ in scored:
        if len(selected) >= n:
            break
        selected.add((mr, nr))

    # Sort: by mr, then nr
    return sorted(selected, key=lambda k: (k[0], k[1]))


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot(data, kernels, out_prefix):
    n = len(kernels)
    if n == 0:
        print("No kernels to plot.")
        return

    labels     = [f'{mr}x{nr}' for mr, nr in kernels]
    xdsl_vals  = [data[(mr, nr)]['xdsl'] for mr, nr in kernels]
    ref_vals   = [data[(mr, nr)]['ref']  for mr, nr in kernels]
    speedups   = [x / r if r > 0 else 0.0 for x, r in zip(xdsl_vals, ref_vals)]

    x     = np.arange(n)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(10, n * 2.2), 6))
    ax2 = ax1.twinx()

    COLOR_XDSL = '#6BC5E8'   # blue
    COLOR_REF  = '#F5A623'   # orange

    bars_xdsl = ax1.bar(x - width/2, xdsl_vals, width, color=COLOR_XDSL, label='xDSL micro-kernels',      zorder=3)
    bars_ref  = ax1.bar(x + width/2, ref_vals,  width, color=COLOR_REF,  label='Reference micro-kernels', zorder=3)

    # Annotate GFLOPS values on bar tops
    for bar, val in zip(bars_xdsl, xdsl_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='#1a1a1a')
    for bar, val in zip(bars_ref, ref_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='#1a1a1a')

    # Speedup ratio line on right axis
    line, = ax2.plot(x, speedups, 'o-', color='black', linewidth=1.8,
                     markersize=6, label='Speedup Ratio', zorder=4)
    for xi, sp in zip(x, speedups):
        ax2.text(xi, sp + 0.015, f'{sp:.2f}x', ha='center', va='bottom',
                 fontsize=11, color='black')

    # Axes labels
    ax1.set_xlabel('Kernel Dimensions (mr x nr)')
    ax1.set_ylabel('Performance (GFLOPS)')
    ax2.set_ylabel('Speedup Ratio (xDSL/Ref)')

    ax1.set_title('Performance Comparison: xDSL vs Reference (Selected Kernels)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ymax = max(max(xdsl_vals), max(ref_vals))
    ax1.set_ylim(0, ymax * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    sp_min = min(speedups)
    sp_max = max(speedups)
    margin = max(0.1, (sp_max - sp_min) * 0.5 + 0.05)
    ax2.set_ylim(sp_min - margin, sp_max + margin)

    # Combined legend (upper left, matching image)
    handles = [line,
               plt.Rectangle((0, 0), 1, 1, color=COLOR_XDSL),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_REF)]
    labels_leg = ['Speedup Ratio', 'xDSL micro-kernels', 'Reference micro-kernels']
    ax1.legend(handles, labels_leg, loc='upper left', fontsize=12)

    fig.tight_layout()

    out_pdf = out_prefix + '_ref_comparison.pdf'
    plt.savefig(out_pdf, dpi=150)
    plt.close()
    print(f"  Saved → {out_pdf}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='xDSL vs Reference micro-kernel comparison plot.')
    ap.add_argument('filename', help='Path to the uK test_gemm output .txt file')
    ap.add_argument('--kernels', default='',
                    help='Comma-separated kernel dims to plot, e.g. 8x8,8x15,16x8,16x16. '
                         'If omitted, auto-selects representative kernels.')
    ap.add_argument('--out', default='',
                    help='Output prefix (default: same dir/basename as input file)')
    args = ap.parse_args()

    if not os.path.exists(args.filename):
        sys.exit(f"ERROR: File not found: {args.filename}")

    data = parse_uk_file(args.filename)
    if not data:
        sys.exit("ERROR: No b0 kernel data found – check that reference column is present.")

    # Select kernels
    if args.kernels:
        kernels = []
        for tok in args.kernels.split(','):
            tok = tok.strip()
            m = re.match(r'^(\d+)[xX](\d+)$', tok)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in data:
                    kernels.append(key)
                else:
                    print(f"  [warn] Kernel {tok} not found in data — skipping")
            else:
                print(f"  [warn] Could not parse kernel '{tok}' — skipping")
        if not kernels:
            sys.exit("ERROR: No valid kernels specified.")
    else:
        kernels = auto_select_kernels(data, n=6)
        print(f"  Auto-selected kernels: {[f'{mr}x{nr}' for mr,nr in kernels]}")

    out_prefix = args.out if args.out else os.path.splitext(args.filename)[0]
    plot(data, kernels, out_prefix)


if __name__ == '__main__':
    main()
