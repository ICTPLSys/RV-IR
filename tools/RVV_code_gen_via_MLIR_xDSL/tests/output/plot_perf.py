import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main():
    ap = argparse.ArgumentParser(description='Plot kc-loop performance heatmap from test_gemm output.')
    ap.add_argument('filename', help='Path to the performance .txt file')
    ap.add_argument('--vlen',     default='',  help='Vector length in bits (e.g. 256)')
    ap.add_argument('--kc',       default='',  help='KC profile used (e.g. 256)')
    ap.add_argument('--families', default='',  help='Kernel families (e.g. 4,4)')
    args = ap.parse_args()

    filename = args.filename

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)

    data = {}

    # Parse the text file — only b0 rows
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if '_b0' in line and 'GFLOP/s' in line:
                parts = line.split()
                if not parts:
                    continue
                kernel_name = parts[0]
                if '_b0' in kernel_name:
                    mr_nr = kernel_name.split('_')[0]
                    if 'x' in mr_nr:
                        try:
                            mr_str, nr_str = mr_nr.split('x')
                            mr = int(mr_str)
                            nr = int(nr_str)
                        except ValueError:
                            continue

                        # Last GFLOP/s value on the line
                        gflops = None
                        for i, token in enumerate(parts):
                            if token == 'GFLOP/s':
                                try:
                                    gflops = float(parts[i-1])
                                except ValueError:
                                    pass

                        if gflops is not None:
                            data[(mr, nr)] = gflops

    if not data:
        print("No valid b0 GFLOP/s data found in the file.")
        sys.exit(1)

    # Best kernel
    best_key  = max(data, key=data.__getitem__)
    best_val  = data[best_key]
    best_mr, best_nr = best_key

    max_mr = max(k[0] for k in data.keys())
    max_nr = max(k[1] for k in data.keys())

    grid = np.zeros((max_nr, max_mr))
    for (mr, nr), gflops in data.items():
        grid[nr-1, mr-1] = gflops

    # Dynamically scale figsize for better readability on large dimension grids
    fig_width = max(10, max_mr * 0.4)
    fig_height = max(10, max_nr * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_title("xDSL micro-kernel Performance", fontsize=12)

    cmap = LinearSegmentedColormap.from_list('rwg', ["red", "white", "green"])
    grid_masked = np.ma.masked_where(grid == 0, grid)
    im = ax.imshow(grid_masked, origin='lower', cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(max_mr))
    ax.set_yticks(np.arange(max_nr))
    ax.set_xticklabels(np.arange(1, max_mr + 1))
    ax.set_yticklabels(np.arange(1, max_nr + 1))
    ax.set_xlabel('mr')
    ax.set_ylabel('nr')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('GFLOPS', rotation=-90, va="bottom")

    vmin, vmax = np.min(grid_masked), np.max(grid_masked)
    vmid = (vmin + vmax) / 2

    # Dynamically scale font size
    font_size = 10
    if max_mr >= 24 or max_nr >= 24:
        font_size = 9
    if max_mr >= 32 or max_nr >= 32:
        font_size = 8

    for i in range(max_nr):
        for j in range(max_mr):
            val = grid[i, j]
            if val > 0:
                text_color = 'white' if abs(val - vmid) > (vmax - vmin) * 0.25 else 'black'
                # Use 1 decimal point to save width, e.g. "12.3" instead of "12.34"
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color=text_color, fontsize=font_size)


    fig.tight_layout()

    out_pdf = os.path.splitext(filename)[0] + "_heatmap.pdf"
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"  Heatmap saved → {out_pdf}")
    print(f"  Best kernel (kc loop): {best_mr}×{best_nr}  =  {best_val:.3f} GFLOP/s")


if __name__ == '__main__':
    main()
