#!/bin/bash
# export_env.sh — 导出当前 Python 环境为 environment.yml
set -e
cd "$(dirname "$0")"

python -c "
import sys, subprocess
result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
pkgs = [l.strip() for l in result.stdout.split('\n') if l.strip() and not l.startswith('#')]
print('name: hag-dta')
print('channels:')
print('  - pytorch')
print('  - pyg')
print('  - conda-forge')
print('dependencies:')
print('  - python=3.8')
print('  - pip')
print('  - numpy')
print('  - pandas')
print('  - scipy')
print('  - scikit-learn')
print('  - networkx')
print('  - matplotlib')
print('  - rdkit')
print('  - pip:')
for p in pkgs:
    if any(k in p.lower() for k in ['torch', 'numpy', 'pandas', 'scipy', 'scikit', 'rdkit', 'networkx', 'matplotlib', 'pillow']):
        pkg_name = p.split('==')[0]
        ver = p.split('==')[1] if '==' in p else ''
        print(f'    - {pkg_name}=={ver}')
" > environment.yml

echo "Done: environment.yml"
