#!/usr/bin/env python3
"""
aggregate_results.py — 汇总所有 random.csv，输出 mean ± std

Usage:
  cd ~/HAG-DTA/code_leo
  python aggregate_results.py

读取 HAG_DTA_OUTPUT_ROOT 下所有 *_random.csv，按数据集聚合成表格。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import OUTPUT_ROOT


def aggregate_dataset(csv_dir, dataset):
    """收集一个数据集的所有 fold × seed 结果，返回 DataFrame"""
    rows = []
    for f in sorted(csv_dir.glob(f'{dataset}_*_random.csv')):
        # 文件名格式: davis_Diff_DTA_GIN_fold0_random.csv
        stem = f.stem  # 'davis_Diff_DTA_GIN_fold0_random'
        parts = stem.replace('_random', '').split('_')
        # parts: ['davis', 'Diff', 'DTA', 'GIN', 'fold0']
        # 找 fold 字段
        fold_id = None
        for p in parts:
            if p.startswith('fold'):
                fold_id = int(p[4:])
                break
        if fold_id is None:
            continue
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            rows.append({**row.to_dict(), 'fold': fold_id})

    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    csv_dir = Path(OUTPUT_ROOT)
    if not csv_dir.exists():
        print(f'OUTPUT_ROOT not found: {csv_dir}')
        print('Set HAG_DTA_OUTPUT_ROOT or check config/paths.py')
        sys.exit(1)

    datasets = ['davis', 'kiba', 'Human', 'Celegans']

    print(f'Scanning {csv_dir} ...')
    print()

    for ds in datasets:
        df = aggregate_dataset(csv_dir, ds)
        if df is None or df.empty:
            print(f'{ds}: no results found')
            print()
            continue

        # 区分回归 vs 分类指标
        if ds in ('davis', 'kiba'):
            metrics = ['mse', 'ci', 'r2']
        else:
            metrics = [c for c in df.columns if c not in ('fold',)]

        # Per-fold mean ± std (across seeds)
        print(f'{"=" * 65}')
        print(f'{ds} — per-fold results (mean ± std across 5 seeds)')
        print(f'{"=" * 65}')
        per_fold = df.groupby('fold')[metrics].agg(['mean', 'std'])
        for fold in sorted(df['fold'].unique()):
            row = per_fold.loc[fold]
            parts = []
            for m in metrics:
                mu = row[(m, 'mean')]
                sigma = row[(m, 'std')]
                parts.append(f'{m}={mu:.4f}±{sigma:.4f}')
            print(f'  fold {fold}:  ' + '  '.join(parts))

        # Overall mean ± std (across folds, using per-fold means)
        print()
        print(f'{"─" * 65}')
        print(f'{ds} — overall (mean ± std across {len(df["fold"].unique())} folds)')
        print(f'{"─" * 65}')
        fold_means = df.groupby('fold')[metrics].mean()
        overall_mean = fold_means.mean()
        overall_std = fold_means.std()
        for m in metrics:
            print(f'  {m}: {overall_mean[m]:.4f} ± {overall_std[m]:.4f}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
