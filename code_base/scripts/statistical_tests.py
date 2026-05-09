#!/usr/bin/env python3
"""
statistical_tests.py — HAG-DTA vs baseline 统计显著性检验 (Welch's t-test)

Usage:
  cd ~/HAG-DTA/code_base
  python scripts/statistical_tests.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config.paths import OUTPUT_ROOT


DAVIS_BASELINES = {
    'KronRLS':      {'mse': (0.379, None), 'ci': (0.871, None), 'r2': (0.407, None)},
    'SimBoost':     {'mse': (0.282, None), 'ci': (0.872, None), 'r2': (0.644, None)},
    'WideDTA':      {'mse': (0.262, 0.009), 'ci': (0.886, 0.003), 'r2': (0.633, 0.011)},
    'DeepDTA':      {'mse': (0.261, 0.007), 'ci': (0.878, 0.004), 'r2': (0.630, 0.017)},
    'MATT_DTI':     {'mse': (0.254, None), 'ci': (0.884, 0.004), 'r2': (0.649, 0.009)},
    'GraphDTA':     {'mse': (0.229, 0.005), 'ci': (0.893, 0.002), 'r2': (0.685, 0.016)},
    'MFR-DTA':      {'mse': (0.221, 0.001), 'ci': (0.905, 0.001), 'r2': (0.705, 0.003)},
    'AttentionDTA': {'mse': (0.216, 0.019), 'ci': (0.893, 0.005), 'r2': (None, None)},
}

KIBA_BASELINES = {
    'KronRLS':      {'mse': (0.411, None), 'ci': (0.782, None), 'r2': (0.342, None)},
    'SimBoost':     {'mse': (0.222, None), 'ci': (0.836, None), 'r2': (0.629, None)},
    'WideDTA':      {'mse': (0.179, 0.008), 'ci': (0.875, 0.001), 'r2': (0.675, 0.004)},
    'DeepDTA':      {'mse': (0.194, 0.008), 'ci': (0.863, 0.002), 'r2': (0.673, 0.019)},
    'MATT_DTI':     {'mse': (0.151, None), 'ci': (0.889, 0.001), 'r2': (0.745, 0.008)},
    'GraphDTA':     {'mse': (0.139, 0.008), 'ci': (0.889, 0.001), 'r2': (0.725, 0.018)},
    'MFR-DTA':      {'mse': (0.136, 0.001), 'ci': (0.898, 0.002), 'r2': (0.789, 0.002)},
    'AttentionDTA': {'mse': (0.155, 0.003), 'ci': (0.882, 0.004), 'r2': (None, None)},
}

HUMAN_BASELINES = {
    'TransformerCPI': {'auroc': (0.973, 0.002), 'precision': (0.916, 0.006), 'recall': (0.925, 0.006)},
    'TrimNet-CNN':    {'auroc': (0.970, None), 'precision': (0.918, None), 'recall': (0.953, None)},
    'GNN-CNN':        {'auroc': (0.970, None), 'precision': (0.923, None), 'recall': (0.918, None)},
    'SAG-DTA':        {'auroc': (0.985, 0.002), 'precision': (0.945, 0.014), 'recall': (0.933, 0.011)},
    'IIFDTI':         {'auroc': (0.984, 0.003), 'precision': (0.946, 0.017), 'recall': (0.947, 0.017)},
    'MolTrans':       {'auroc': (0.974, 0.002), 'precision': (0.955, 0.012), 'recall': (0.933, 0.022)},
}

CELEGANS_BASELINES = {
    'TransformerCPI': {'auroc': (0.988, 0.002), 'precision': (0.952, 0.006), 'recall': (0.953, 0.005)},
    'TrimNet-CNN':    {'auroc': (0.987, None), 'precision': (0.946, None), 'recall': (0.945, None)},
    'GNN-CNN':        {'auroc': (0.978, None), 'precision': (0.938, None), 'recall': (0.929, None)},
    'IIFDTI':         {'auroc': (0.991, 0.002), 'precision': (0.954, 0.010), 'recall': (0.971, 0.011)},
    'MolTrans':       {'auroc': (0.982, 0.003), 'precision': (0.971, 0.007), 'recall': (0.963, 0.012)},
}


def load_hag_results(csv_dir, dataset, model='Diff_DTA_GIN'):
    candidates = [
        csv_dir / f'{dataset}_{model}_random.csv',
        csv_dir / f'{dataset}_{model}_random.csv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return {col: df[col].dropna().tolist() for col in df.columns}

    legacy_files = sorted(csv_dir.glob(f'{dataset}_{model}_fold*_random.csv'))
    if legacy_files:
        merged = {}
        for path in legacy_files:
            df = pd.read_csv(path)
            for col in df.columns:
                merged.setdefault(col, []).extend(df[col].dropna().tolist())
        return merged

    return {}


def welch_t_test(hag_values, baseline_mean, baseline_std, better='higher'):
    if baseline_mean is None:
        return None

    hag_values = np.asarray(hag_values, dtype=float)
    hag_mean = np.mean(hag_values)
    hag_std = np.std(hag_values, ddof=1)
    n = len(hag_values)

    if baseline_std is None:
        t_stat, p_value = stats.ttest_1samp(hag_values, baseline_mean)
    else:
        baseline_n = 5
        se = np.sqrt(hag_std ** 2 / n + baseline_std ** 2 / baseline_n)
        t_stat = (hag_mean - baseline_mean) / se
        df_num = (hag_std ** 2 / n + baseline_std ** 2 / baseline_n) ** 2
        df_den = ((hag_std ** 2 / n) ** 2 / (n - 1) + (baseline_std ** 2 / baseline_n) ** 2 / (baseline_n - 1))
        df = df_num / df_den
        p_value = 2 * stats.t.sf(abs(t_stat), df)

    if better == 'lower':
        t_stat = -t_stat

    return {'t_stat': t_stat, 'p_value': p_value, 'hag_mean': hag_mean, 'hag_std': hag_std}


def main():
    csv_dir = Path(OUTPUT_ROOT)
    if not csv_dir.exists():
        print(f'ERROR: OUTPUT_ROOT not found: {csv_dir}')
        sys.exit(1)

    print('=' * 72)
    print("STATISTICAL SIGNIFICANCE TESTS — HAG-DTA vs Baselines")
    print('=' * 72)
    print()

    configs = [
        ('davis', 'mse', 'lower', DAVIS_BASELINES),
        ('davis', 'ci', 'higher', DAVIS_BASELINES),
        ('davis', 'r2', 'higher', DAVIS_BASELINES),
        ('kiba', 'mse', 'lower', KIBA_BASELINES),
        ('kiba', 'ci', 'higher', KIBA_BASELINES),
        ('kiba', 'r2', 'higher', KIBA_BASELINES),
        ('Human', 'AUROC', 'higher', HUMAN_BASELINES),
        ('Human', 'Precision', 'higher', HUMAN_BASELINES),
        ('Human', 'Recall', 'higher', HUMAN_BASELINES),
        ('Celegans', 'AUROC', 'higher', CELEGANS_BASELINES),
        ('Celegans', 'Precision', 'higher', CELEGANS_BASELINES),
        ('Celegans', 'Recall', 'higher', CELEGANS_BASELINES),
    ]

    for dataset, metric, better, baselines in configs:
        all_vals = load_hag_results(csv_dir, dataset)
        if metric not in all_vals or not all_vals[metric]:
            print(f'{dataset} / {metric}: no HAG results yet')
            continue

        hag_arr = np.array(all_vals[metric], dtype=float)
        print(f'--- {dataset} — {metric} ({"↑" if better == "higher" else "↓"}) ---')
        print(f'  HAG-DTA: {hag_arr.mean():.4f} ± {hag_arr.std(ddof=1):.4f} (n={len(hag_arr)})')
        print(f'  {"Model":<16} {"Baseline":>14} {"t-stat":>8} {"p-value":>9}  {"Significant?"}')
        print(f'  {"-" * 16} {"-" * 14} {"-" * 8} {"-" * 9}  {"-" * 13}')

        for name, vals in baselines.items():
            baseline_mean, baseline_std = vals.get(metric.lower(), (None, None))
            result = welch_t_test(hag_arr, baseline_mean, baseline_std, better)
            if result is None:
                continue
            sig = '✅ YES' if result['p_value'] < 0.05 else ('⚠️  p<0.1' if result['p_value'] < 0.10 else '   no')
            baseline_str = f'{baseline_mean:.4f}' + (f'±{baseline_std:.4f}' if baseline_std else '')
            print(f'  {name:<16} {baseline_str:>14} {result["t_stat"]:>8.2f} {result["p_value"]:>9.4f}  {sig}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
