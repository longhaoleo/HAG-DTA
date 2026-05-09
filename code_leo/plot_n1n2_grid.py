#!/usr/bin/env python3
"""
plot_n1n2_grid.py — n1/n2 grid search 3D visualization
Creates 3D surface plots showing joint effect of n1 and n2 on key metrics.
Output: n1n2_grid_davis.png, n1n2_grid_human.png
"""

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_ROOT = os.environ.get('HAG_DTA_OUTPUT_ROOT',
    os.path.expanduser('~/HAG-DTA/OUTPUT'))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../LOG')
os.makedirs(OUT_DIR, exist_ok=True)

ALL_COMBOS = [(n1, n2) for n1 in range(4, 9) for n2 in range(2, 5) if n2 < n1]
DAVIS_METRICS = ['mse', 'ci', 'r2']
HUMAN_METRICS = ['AUROC', 'AUPRC', 'Precision', 'Recall']

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ── Parse Davis logs ────────────────────────────────────────────────
def parse_davis():
    sensitivity_dir = os.path.join(OUTPUT_ROOT, 'sensitivity')
    results = []
    for f in sorted(os.listdir(sensitivity_dir)):
        m = re.match(r'n1_(\d+)_n2_(\d+)\.log', f)
        if not m:
            continue
        n1, n2 = int(m.group(1)), int(m.group(2))
        with open(os.path.join(sensitivity_dir, f)) as fp:
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
        # fallback: CSV output line
        if not metrics:
            for line in reversed(content.split('\n')):
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
        if metrics:
            metrics['n1'] = n1
            metrics['n2'] = n2
            results.append(metrics)
    df = pd.DataFrame(results)
    missing = set(ALL_COMBOS) - set(zip(df['n1'], df['n2']))
    if missing:
        print(f'[Davis] Missing combos: {missing}')
    return df


# ── Parse Human logs ────────────────────────────────────────────────
def parse_human():
    sensitivity_dir = os.path.join(OUTPUT_ROOT, 'sensitivity_human')
    results = []
    for f in sorted(os.listdir(sensitivity_dir)):
        m = re.match(r'n1_(\d+)_n2_(\d+)\.log', f)
        if not m:
            continue
        n1, n2 = int(m.group(1)), int(m.group(2))
        with open(os.path.join(sensitivity_dir, f), errors='ignore') as fp:
            lines = fp.readlines()
        # Find last line with best_AUROC
        best_line = None
        for line in lines:
            if 'best_AUROC' in line and 'best_AUPRC' in line:
                best_line = line
        if best_line is None:
            continue
        # Format: "best_AUROC, best_AUPRC, best_precision, best_recall: 0.9889 0.9894 0.9386 0.9565"
        parts = best_line.split(':')
        if len(parts) >= 2:
            nums = parts[-1].strip().split()
            if len(nums) >= 2:
                entry = {'n1': n1, 'n2': n2,
                         'AUROC': float(nums[0]),
                         'AUPRC': float(nums[1])}
                if len(nums) >= 3:
                    entry['Precision'] = float(nums[2])
                if len(nums) >= 4:
                    entry['Recall'] = float(nums[3])
                results.append(entry)
    df = pd.DataFrame(results)
    missing = set(ALL_COMBOS) - set(zip(df['n1'], df['n2']))
    if missing:
        print(f'[Human] Missing combos: {missing}')
    return df


# ── Build grid for 3D surface ───────────────────────────────────────
def build_grid(df, metric):
    """Build dense grid matrices for 3D surface plot."""
    n1_vals = sorted(df['n1'].unique())
    n2_vals = sorted(df['n2'].unique())
    Z = np.full((len(n1_vals), len(n2_vals)), np.nan)
    for _, row in df.iterrows():
        if metric not in row:
            continue
        i = n1_vals.index(row['n1'])
        j = n2_vals.index(row['n2'])
        Z[i, j] = row[metric]
    N1, N2 = np.meshgrid(n2_vals, n1_vals)  # meshgrid expects (cols, rows)
    return N1, N2, Z


# ── 3D surface plot ─────────────────────────────────────────────────
def plot_3d_grid(N1, N2, Z, metric_name, dataset_name, cmap_name='viridis'):
    """Single 3D surface subplot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(N1, N2, Z, cmap=cm.get_cmap(cmap_name),
                           edgecolor='k', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('n2', labelpad=10)
    ax.set_ylabel('n1', labelpad=10)
    ax.set_zlabel(metric_name, labelpad=10)
    ax.set_title(f'{dataset_name} — {metric_name} vs (n1, n2)', fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.view_init(elev=25, azim=-60)
    return fig


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print('Parsing Davis sensitivity results ...')
    df_davis = parse_davis()
    print(f'  {len(df_davis)} results loaded')

    print('Parsing Human sensitivity results ...')
    df_human = parse_human()
    print(f'  {len(df_human)} results loaded')

    # ── Davis: 3 metrics as separate 3D plots ────────────────────────
    for metric in DAVIS_METRICS:
        if metric not in df_davis.columns:
            continue
        N1, N2, Z = build_grid(df_davis, metric)
        fig = plot_3d_grid(N1, N2, Z, metric.upper(), 'Davis')
        fname = f'n1n2_davis_{metric}.png'
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {fname}')

    # ── Davis: combined figure (3 subplots) ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                             subplot_kw={'projection': '3d'})
    fig.suptitle('Davis — n1/n2 Grid Search', fontsize=14, fontweight='bold')
    for idx, metric in enumerate(DAVIS_METRICS):
        ax = axes[idx]
        N1, N2, Z = build_grid(df_davis, metric)
        surf = ax.plot_surface(N1, N2, Z, cmap=cm.get_cmap('viridis'),
                               edgecolor='k', linewidth=0.3, alpha=0.85)
        ax.set_xlabel('n2')
        ax.set_ylabel('n1')
        ax.set_zlabel(metric.upper())
        ax.set_title(metric.upper())
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        ax.view_init(elev=25, azim=-60)
    fig.savefig(os.path.join(OUT_DIR, 'n1n2_grid_davis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved n1n2_grid_davis.png (combined)')

    # ── Human: 4 metrics as separate 3D plots ────────────────────────
    for metric in HUMAN_METRICS:
        if metric not in df_human.columns:
            continue
        N1, N2, Z = build_grid(df_human, metric)
        fig = plot_3d_grid(N1, N2, Z, metric, 'Human')
        fname = f'n1n2_human_{metric}.png'
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {fname}')

    # ── Human: combined figure (4 subplots) ──────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(28, 6),
                             subplot_kw={'projection': '3d'})
    fig.suptitle('Human — n1/n2 Grid Search', fontsize=14, fontweight='bold')
    for idx, metric in enumerate(HUMAN_METRICS):
        ax = axes[idx]
        N1, N2, Z = build_grid(df_human, metric)
        surf = ax.plot_surface(N1, N2, Z, cmap=cm.get_cmap('viridis'),
                               edgecolor='k', linewidth=0.3, alpha=0.85)
        ax.set_xlabel('n2')
        ax.set_ylabel('n1')
        ax.set_zlabel(metric)
        ax.set_title(metric)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        ax.view_init(elev=25, azim=-60)
    fig.savefig(os.path.join(OUT_DIR, 'n1n2_grid_human.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved n1n2_grid_human.png (combined)')

    # ── Print summary tables ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Davis n1/n2 Grid — Best 5 by CI')
    print('=' * 60)
    best_davis = df_davis.nlargest(5, 'ci')
    for _, r in best_davis.iterrows():
        print(f"  ({int(r['n1'])},{int(r['n2'])}):  MSE={r['mse']:.4f}  CI={r['ci']:.4f}  rm2={r['r2']:.4f}")

    print('\n' + '=' * 60)
    print('Human n1/n2 Grid — Best 5 by AUROC')
    print('=' * 60)
    best_human = df_human.nlargest(5, 'AUROC')
    for _, r in best_human.iterrows():
        print(f"  ({int(r['n1'])},{int(r['n2'])}):  AUROC={r['AUROC']:.4f}  AUPRC={r['AUPRC']:.4f}  "
              f"Prec={r.get('Precision',-1):.4f}  Rec={r.get('Recall',-1):.4f}")

    print(f'\nAll figures saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
