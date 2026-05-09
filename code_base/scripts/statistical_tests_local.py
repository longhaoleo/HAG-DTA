#!/usr/bin/env python3
"""
statistical_tests_local.py — 本地两组实验结果的显著性比较

默认用于比较：HAG-DTA vs TransformerCPI（Human / C.elegans）

Usage:
  cd ~/HAG-DTA/code_base
  python scripts/statistical_tests_local.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config.paths import OUTPUT_ROOT


def load_result(csv_dir, filename):
    path = csv_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def welch_t_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
    return t_stat, p_value


def compare_pair(csv_dir, dataset, left_file, right_file, metrics):
    left = load_result(csv_dir, left_file)
    right = load_result(csv_dir, right_file)
    if left is None or right is None:
        print(f'{dataset}: missing file(s) -> {left_file} / {right_file}')
        return

    print('=' * 72)
    print(f'{dataset}: {left_file}  vs  {right_file}')
    print('=' * 72)
    print(f'{"Metric":<12} {"Left mean±std":<24} {"Right mean±std":<24} {"t-stat":>8} {"p-value":>10}')
    print('-' * 72)
    for metric in metrics:
        if metric not in left.columns or metric not in right.columns:
            continue
        left_vals = left[metric].dropna().to_numpy()
        right_vals = right[metric].dropna().to_numpy()
        t_stat, p_value = welch_t_test(left_vals, right_vals)
        print(
            f'{metric:<12} '
            f'{left_vals.mean():.4f}±{left_vals.std(ddof=1):.4f}      '
            f'{right_vals.mean():.4f}±{right_vals.std(ddof=1):.4f}      '
            f'{t_stat:>8.2f} {p_value:>10.4f}'
        )
    print()


if __name__ == '__main__':
    csv_dir = Path(OUTPUT_ROOT)
    if not csv_dir.exists():
        print(f'OUTPUT_ROOT not found: {csv_dir}')
        sys.exit(1)

    compare_pair(
        csv_dir,
        'Human',
        'Human_Diff_DTA_GIN_random.csv',
        'Human_TransformerCPI_random.csv',
        ['AUROC', 'AUPRC', 'Precision', 'Recall'],
    )
    compare_pair(
        csv_dir,
        'Celegans',
        'Celegans_Diff_DTA_GIN_random.csv',
        'Celegans_TransformerCPI_random.csv',
        ['AUROC', 'AUPRC', 'Precision', 'Recall'],
    )
