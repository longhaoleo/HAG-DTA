#!/usr/bin/env python3
"""
plot_mmd_beta.py — MMD loss coefficient ablation visualization
Line plots showing effect of beta on MSE, CI, rm2 for Davis + GIN.
Output: mmd_beta_davis.png
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_ROOT = os.environ.get('HAG_DTA_OUTPUT_ROOT',
    os.path.expanduser('~/HAG-DTA/OUTPUT'))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../REPORT2')
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0, 0.01, 0.05, 0.1, 0.5, 1.0]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ── Parse MMD logs ──────────────────────────────────────────────────
def parse_mmd():
    base = os.path.join(OUTPUT_ROOT, 'sensitivity_mmd')
    summary = os.path.join(base, 'mmd_beta_summary.csv')
    if os.path.exists(summary):
        df = pd.read_csv(summary)
        grouped = df.groupby('beta', as_index=False).agg({
            'mse': ['mean', 'std'],
            'ci': ['mean', 'std'],
            'rm2': ['mean', 'std'],
        })
        grouped.columns = [
            'beta', 'mse', 'mse_std', 'ci', 'ci_std', 'r2', 'r2_std'
        ]
        return grouped.sort_values('beta').to_dict('records')

    results = []
    for beta in BETAS:
        candidates = [
            os.path.join(base, f'mmd_beta_{beta}.log'),
            os.path.join(base, f'mmd_beta_{beta}_seed_100.log'),
        ]
        fpath = next((p for p in candidates if os.path.exists(p)), None)
        if fpath is None:
            continue
        with open(fpath, errors='ignore') as fp:
            content = fp.read()
        metrics = {}
        for line in content.split('\n'):
            if 'final test metrics' in line:
                mse_m = re.search(r'mse=\s*([\d.]+)', line)
                ci_m = re.search(r'ci=\s*([\d.]+)', line)
                r2_m = re.search(r'rm2=\s*([\d.]+)', line)
                if mse_m and ci_m and r2_m:
                    metrics = {'mse': float(mse_m.group(1)),
                               'ci': float(ci_m.group(1)),
                               'r2': float(r2_m.group(1))}
                break
        if not metrics:
            for line in content.split('\n'):
                if 'MSE=' in line and 'CI=' in line:
                    mse_m = re.search(r'MSE=([\d.]+)', line)
                    ci_m = re.search(r'CI=([\d.]+)', line)
                    r2_m = re.search(r'rm.?\s*=\s*([\d.]+)', line)
                    if mse_m and ci_m:
                        metrics = {'mse': float(mse_m.group(1)),
                                   'ci': float(ci_m.group(1))}
                        if r2_m:
                            metrics['r2'] = float(r2_m.group(1))
                    break
        if not metrics:
            for line in content.split('\n'):
                if 'final test | epoch' in line:
                    mse_m = re.search(r'mse=([\d.]+)', line)
                    ci_m = re.search(r'ci=([\d.]+)', line)
                    r2_m = re.search(r'rm2=([\d.]+)', line)
                    if mse_m and ci_m and r2_m:
                        metrics = {'mse': float(mse_m.group(1)),
                                   'ci': float(ci_m.group(1)),
                                   'r2': float(r2_m.group(1))}
        if metrics:
            metrics['beta'] = beta
            results.append(metrics)
    return sorted(results, key=lambda x: x['beta'])


# ── Plot ─────────────────────────────────────────────────────────────
def main():
    results = parse_mmd()
    print(f'Loaded {len(results)} MMD ablation results')

    betas = [r['beta'] for r in results]
    mse_vals = [r['mse'] for r in results]
    ci_vals = [r['ci'] for r in results]
    r2_vals = [r['r2'] for r in results]

    # Default beta=0.05 marker
    default_idx = betas.index(0.05)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle('MMD Loss Coefficient Ablation — Davis + GIN', fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    labels = ['MSE (lower is better)', 'CI (higher is better)', 'rm² (higher is better)']
    y_vals = [mse_vals, ci_vals, r2_vals]
    markers = ['o', 's', '^']

    for idx, (ax, y, color, label, marker) in enumerate(zip(axes, y_vals, colors, labels, markers)):
        ax.plot(betas, y, '-o', color=color, linewidth=2, markersize=8, marker=marker,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot(betas[default_idx], y[default_idx], 'D', color='red', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.set_xlabel('β')
        ax.set_ylabel(['MSE', 'CI', 'rm²'][idx])
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('symlog', linthresh=0.01)

    # Add value annotations
    for idx, (ax, y) in enumerate(zip(axes, y_vals)):
        for i in range(len(betas)):
            offset = 0.001 if idx == 0 else 0.0005
            ax.annotate(f'{y[i]:.4f}', (betas[i], y[i]),
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=7, alpha=0.8)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, 'mmd_beta_davis.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')

    # Print summary table
    print('\n' + '=' * 60)
    print('MMD Beta Ablation Summary — Davis + GIN')
    print('=' * 60)
    print(f'{"β":<10} {"MSE":<12} {"CI":<12} {"rm²":<12}')
    print('-' * 46)
    for r in results:
        print(f'{r["beta"]:<10} {r["mse"]:<12.4f} {r["ci"]:<12.4f} {r["r2"]:<12.4f}')
    print('-' * 46)
    # Best row
    best_mse = min(results, key=lambda x: x['mse'])
    best_ci = max(results, key=lambda x: x['ci'])
    print(f'Best MSE:  β={best_mse["beta"]} ({best_mse["mse"]:.4f})')
    print(f'Best CI:   β={best_ci["beta"]} ({best_ci["ci"]:.4f})')
    print(f'Default:   β=0.05 (MSE={results[default_idx]["mse"]:.4f}, CI={results[default_idx]["ci"]:.4f})')


if __name__ == '__main__':
    main()
