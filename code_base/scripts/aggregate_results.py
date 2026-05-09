#!/usr/bin/env python3
"""
aggregate_results.py — 汇总所有 random.csv，输出 mean ± std

Usage:
  cd ~/HAG-DTA/code_base
  python scripts/aggregate_results.py
"""
from pathlib import Path
import sys

import pandas as pd

from config.paths import OUTPUT_ROOT


def find_result_files(csv_dir, dataset):
    return sorted(csv_dir.glob(f'{dataset}_*_random.csv'))


def parse_model_name(dataset, path):
    stem = path.stem
    prefix = f'{dataset}_'
    suffix = '_random'
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return stem
    return stem[len(prefix):-len(suffix)]


def summarize_file(path):
    df = pd.read_csv(path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    return {col: (df[col].mean(), df[col].std(ddof=1)) for col in numeric_cols}


def main():
    csv_dir = Path(OUTPUT_ROOT)
    if not csv_dir.exists():
        print(f'OUTPUT_ROOT not found: {csv_dir}')
        print('Set HAG_DTA_OUTPUT_ROOT or check config/paths.py')
        sys.exit(1)

    datasets = ['davis', 'kiba', 'Human', 'Celegans']
    print(f'Scanning {csv_dir} ...\n')

    for dataset in datasets:
        files = find_result_files(csv_dir, dataset)
        if not files:
            print(f'{dataset}: no results found\n')
            continue

        print('=' * 72)
        print(dataset)
        print('=' * 72)
        for path in files:
            model_name = parse_model_name(dataset, path)
            summary = summarize_file(path)
            if summary is None:
                continue
            parts = []
            for metric, (mean_value, std_value) in summary.items():
                parts.append(f'{metric}={mean_value:.4f}±{std_value:.4f}')
            print(f'{model_name:<28} ' + '  '.join(parts))
        print()

    print('Done.')


if __name__ == '__main__':
    main()
