#!/usr/bin/env python3
"""
sensitivity_n1n2.py — n1/n2 敏感性分析

Usage:
  cd ~/HAG-DTA/code_leo
  python sensitivity_n1n2.py

在 Davis 数据集上用 GIN 模型测试不同 (n1, n2) 组合，
每组跑 1 fold + 1 seed，输出对比表格。
"""
import os
import subprocess
import sys
from pathlib import Path

from config.paths import OUTPUT_ROOT, ensure_runtime_dirs

ensure_runtime_dirs()

# ── 搜索空间 ──────────────────────────────────────
N1_VALUES = [4, 5, 6, 7, 8]
N2_VALUES = [2, 3, 4]
DATASET_ID = 0    # Davis
MODEL_ID = 0      # GIN
FOLD_ID = 0
SEED_OVERRIDE = 100  # 只用 1 个种子加速

combos = [(n1, n2) for n1 in N1_VALUES for n2 in N2_VALUES if n2 < n1]

results = []

for n1, n2 in combos:
    tag = f'n1_{n1}_n2_{n2}'
    print(f'{"="*60}')
    print(f'Running: n1={n1}, n2={n2}')
    print(f'{"="*60}')

    env = os.environ.copy()
    env['HAG_DTA_N1'] = str(n1)
    env['HAG_DTA_N2'] = str(n2)
    env['HAG_DTA_SEED_OVERRIDE'] = str(SEED_OVERRIDE)

    # 用临时 seed 覆盖
    cmd = [
        sys.executable, '-c',
        f'''
import os, sys, subprocess
# 临时修改 config/training.py 的 SEEDS
import config.training as ct
ct.SEEDS = [{SEED_OVERRIDE}]
# 设置环境变量让训练脚本读取 n1/n2
os.environ['HAG_DTA_N1'] = '{n1}'
os.environ['HAG_DTA_N2'] = '{n2}'
# 直接 import 并运行 training
sys.argv = ['training_davis_kiba.py', '{DATASET_ID}', '{MODEL_ID}', '{FOLD_ID}']
exec(open('training_davis_kiba.py').read())
'''
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f'FAILED: n1={n1}, n2={n2}')
        print(result.stderr[-500:])
        results.append({'n1': n1, 'n2': n2, 'status': 'FAILED'})
        continue

    # 读取 random.csv 获取最终指标
    import pandas as pd
    csv_path = Path(OUTPUT_ROOT) / f'davis_Diff_DTA_GIN_fold{FOLD_ID}_random.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        row = {'n1': n1, 'n2': n2, 'status': 'OK',
               'mse': df['mse'].iloc[-1],
               'ci': df['ci'].iloc[-1],
               'r2': df['r2'].iloc[-1]}
    else:
        row = {'n1': n1, 'n2': n2, 'status': 'NO_CSV'}
    results.append(row)
    print(f'  Result: {row}')
    print()

# ── 输出汇总表格 ──────────────────────────────────
print()
print('='*65)
print('SENSITIVITY ANALYSIS RESULTS — Davis + GIN + fold0')
print('='*65)
print(f'{"n1":>5} {"n2":>5} {"MSE":>10} {"CI":>8} {"rm²":>8}')
print('-'*40)
for r in results:
    if r['status'] == 'OK':
        print(f'{r["n1"]:>5} {r["n2"]:>5} {r["mse"]:>10.4f} {r["ci"]:>8.4f} {r["r2"]:>8.4f}')
    else:
        print(f'{r["n1"]:>5} {r["n2"]:>5} {"FAILED":>10}')

print()
best = max((r for r in results if r['status'] == 'OK'), key=lambda x: x['ci'], default=None)
if best:
    print(f'Best CI:  n1={best["n1"]}, n2={best["n2"]} → CI={best["ci"]:.4f}')
best_mse = min((r for r in results if r['status'] == 'OK'), key=lambda x: x['mse'], default=None)
if best_mse:
    print(f'Best MSE: n1={best_mse["n1"]}, n2={best_mse["n2"]} → MSE={best_mse["mse"]:.4f}')
