#!/usr/bin/env python3
"""
plot_mmd_beta.py - MMD beta sensitivity visualization.
Reads Davis and Human beta summaries and writes one figure per dataset.
"""

import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_ROOT = os.environ.get('HAG_DTA_OUTPUT_ROOT', '/root/autodl-tmp/HAG-DTA-runs')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../REPORT2')
os.makedirs(OUT_DIR, exist_ok=True)

BETAS = [0, 0.01, 0.05, 0.1, 0.5, 1.0]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


DATASETS = [
    {
        'name': 'Davis',
        'slug': 'davis',
        'summary': os.path.join(OUTPUT_ROOT, 'sensitivity_mmd_davis', 'mmd_beta_davis_summary.csv'),
        'metrics': [('mse', 'MSE', 'lower'), ('ci', 'CI', 'higher'), ('rm2', 'rm2', 'higher')],
    },
    {
        'name': 'Human',
        'slug': 'human',
        'summary': os.path.join(OUTPUT_ROOT, 'sensitivity_mmd_human', 'mmd_beta_human_summary.csv'),
        'metrics': [('AUROC', 'AUROC', 'higher'), ('AUPRC', 'AUPRC', 'higher'),
                    ('Precision', 'Precision', 'higher'), ('Recall', 'Recall', 'higher')],
    },
]


def load_summary(path, metrics):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    columns = ['beta'] + [name for name, _, _ in metrics]
    df = df[columns].copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['beta'])

    metric_names = [name for name, _, _ in metrics]
    grouped = df.groupby('beta')[metric_names].agg(['mean', 'std']).reset_index()

    flattened = ['beta']
    for name, _, _ in metrics:
        flattened.extend([name, f'{name}_std'])
    grouped.columns = flattened
    return grouped.sort_values('beta')


def plot_dataset(spec, df):
    metrics = spec['metrics']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle(f'MMD beta sensitivity - {spec["name"]} + HAG-DTA GIN',
                 fontsize=14, fontweight='bold')

    colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']
    betas = df['beta'].tolist()
    default_rows = df.index[df['beta'] == 0.05].tolist()

    for ax, (metric, label, direction), color in zip(axes, metrics, colors):
        y = df[metric].tolist()
        yerr = df.get(f'{metric}_std', pd.Series([0] * len(df))).fillna(0).tolist()
        ax.errorbar(betas, y, yerr=yerr, marker='o', linewidth=2, capsize=3,
                    color=color, markeredgecolor='white', markeredgewidth=1)
        if default_rows:
            idx = default_rows[0]
            ax.plot(df.loc[idx, 'beta'], df.loc[idx, metric], 'D', color='red',
                    markersize=9, markeredgecolor='white', markeredgewidth=1.2)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(label)
        ax.set_title(f'{label} ({"lower" if direction == "lower" else "higher"} is better)')
        ax.set_xscale('symlog', linthresh=0.01)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'mmd_beta_{spec["slug"]}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')


def print_summary(spec, df):
    print('\n' + '=' * 70)
    print(f'MMD beta sensitivity summary - {spec["name"]} + HAG-DTA GIN')
    print('=' * 70)
    cols = ['beta'] + [name for name, _, _ in spec['metrics']]
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def main():
    loaded = 0
    for spec in DATASETS:
        df = load_summary(spec['summary'], spec['metrics'])
        if df is None:
            print(f'Skip {spec["name"]}: summary not found or empty: {spec["summary"]}')
            continue
        loaded += 1
        print_summary(spec, df)
        plot_dataset(spec, df)

    if loaded == 0:
        raise SystemExit('No MMD beta summaries found.')


if __name__ == '__main__':
    main()
