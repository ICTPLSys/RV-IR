#!/usr/bin/env python3
import sys
import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 14})

def parse_and_plot(input_file, out_prefix):
    # Dictionaries to store layer data
    # format: layer_id -> (M, N, K, gflops)
    xdsl_data = {}
    oblas_data = {}
    
    # Regex for xDSL Best Kernels
    # Example: | Layer  1 (M=  16, N=2708, K=1433):   1.68e+00 GFLOPS  =>  Kernel [16x 3]
    xdsl_re = re.compile(r'\|\s*Layer\s+(\d+)\s*\(M=\s*(\d+),\s*N=\s*(\d+),\s*K=\s*(\d+)\):\s+([\d.e+-]+)\s+GFLOPS')
    
    # Regex for OpenBLAS
    # Example: 1                          16     2708     1433   | 62764.88 us |     1.98 GFLOPS
    oblas_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\|\s*[\d.]+\s*us\s*\|\s*([\d.]+)\s*GFLOPS')
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                # check xDSL match
                m_x = xdsl_re.search(line)
                if m_x:
                    l_id = int(m_x.group(1))
                    m, n, k = int(m_x.group(2)), int(m_x.group(3)), int(m_x.group(4))
                    gflops = float(m_x.group(5))
                    xdsl_data[l_id] = {'m': m, 'n': n, 'k': k, 'gflops': gflops}
                    continue
                
                # check OBLAS match
                m_o = oblas_re.match(line)
                if m_o:
                    l_id = int(m_o.group(1))
                    m, n, k = int(m_o.group(2)), int(m_o.group(3)), int(m_o.group(4))
                    gflops = float(m_o.group(5))
                    oblas_data[l_id] = {'m': m, 'n': n, 'k': k, 'gflops': gflops}
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    # Intersection of layer IDs
    layer_ids = sorted(list(set(xdsl_data.keys()).intersection(set(oblas_data.keys()))))
    
    if not layer_ids:
        print("Error: Could not extract matching layers for xDSL and OpenBLAS from the file.")
        sys.exit(1)
        
    xdsl_gflops = [xdsl_data[lid]['gflops'] for lid in layer_ids]
    oblas_gflops = [oblas_data[lid]['gflops'] for lid in layer_ids]
    labels = [f"L{lid}" for lid in layer_ids]
    
    x = np.arange(len(layer_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(layer_ids) * 2), 5))
    
    COLOR_OBLAS = '#888888'
    COLOR_XDSL  = '#6BC5E8'
    
    bars_ob = ax.bar(x - width/2, oblas_gflops, width, color=COLOR_OBLAS, label='OpenBLAS (RVV)')
    bars_xl = ax.bar(x + width/2, xdsl_gflops, width, color=COLOR_XDSL, label='xDSL micro-kernels')
    
    # Annotate values on top of bars
    for bar, val in zip(bars_ob, oblas_gflops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='#333333')
    for bar, val in zip(bars_xl, xdsl_gflops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='#1a6f9c')

    model_name = os.path.splitext(os.path.basename(input_file))[0].upper()

    ax.set_xlabel('Layer')
    ax.set_ylabel('GFLOPS')
    ax.set_title(f'{model_name} – Performance Comparison: xDSL vs OpenBLAS')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 13)
    ax.set_yticks(np.arange(0, 13, 2))
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    
    out_pdf = out_prefix + '.pdf'
    plt.savefig(out_pdf, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_pdf}")
    
    out_txt = out_prefix + '.txt'
    with open(out_txt, 'w') as f:
        f.write(f"# Performance Comparison: xDSL vs OpenBLAS\n")
        f.write(f"# Extracted from: {input_file}\n#\n")
        f.write(f"# {'Layer':<6} {'M':>6} {'N':>6} {'K':>6}  {'xDSL_GFLOPS':>12}  {'OpenBLAS_GFLOPS':>15}  {'Ratio':>7}\n")
        f.write(f"# {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*12}  {'-'*15}  {'-'*7}\n")
        
        for lid in layer_ids:
            xd = xdsl_data[lid]
            od = oblas_data[lid]
            ratio = xd['gflops'] / od['gflops'] if od['gflops'] > 0 else float('nan')
            f.write(f"  L{lid:<5} {xd['m']:>6} {xd['n']:>6} {xd['k']:>6}  {xd['gflops']:>12.3f}  {od['gflops']:>15.3f}  {ratio:>7.3f}\n")
            
    print(f"Saved text summary to: {out_txt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse single file with both xDSL sweep and OpenBLAS results')
    parser.add_argument('input', help='Input text file (e.g. gnn.txt)')
    parser.add_argument('--out', default='', help='Output prefix')
    args = parser.parse_args()
    
    if not args.out:
        args.out = os.path.splitext(args.input)[0] + "_perf"
        
    parse_and_plot(args.input, args.out)
