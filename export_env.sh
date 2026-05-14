#!/usr/bin/env bash
# export_env.sh
# 导出当前 Python 环境为 environment.yml

set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${1:-hag-dta}"

python - <<'PY' > environment.yml
import sys
import subprocess
import re
import importlib.util

env_name = "hag-dta"

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

def get_version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "")
    except Exception:
        return ""

def normalize_conda_version(version):
    if not version:
        return ""
    version = version.split("+", 1)[0]
    m = re.match(r"^(\d+\.\d+)", version)
    return m.group(1) + ".*" if m else version

torch_version = normalize_conda_version(get_version("torch"))
pyg_version = normalize_conda_version(get_version("torch_geometric"))

result = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    capture_output=True,
    text=True,
    check=True,
)

exclude_prefixes = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "networkx",
    "matplotlib",
    "rdkit",
    "rdkit-pypi",
    "pillow",
    "pil",
    "torch",
    "torchvision",
    "torchaudio",
    "torch-geometric",
    "pyg",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
}

pip_packages = []

for line in result.stdout.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    name = re.split(r"==|>=|<=|~=|!=|>|<|@", line, maxsplit=1)[0].strip().lower()
    name = name.replace("_", "-")

    if name in exclude_prefixes:
        continue

    if line == "aiosignal==1.3.1x":
        line = "aiosignal==1.3.1"

    pip_packages.append(line)

print(f"name: {env_name}")
print("channels:")
print("  - pytorch")
print("  - pyg")
print("  - conda-forge")
print("dependencies:")
print(f"  - python={python_version}")
print("  - pip")
print("  - numpy")
print("  - pandas")
print("  - scipy")
print("  - scikit-learn")
print("  - networkx")
print("  - matplotlib")
print("  - rdkit")

if torch_version:
    print(f"  - pytorch={torch_version}")
else:
    print("  - pytorch")

print("  - torchvision")
print("  - torchaudio")

if pyg_version:
    print(f"  - pyg={pyg_version}")
else:
    print("  - pyg")

if pip_packages:
    print("  - pip:")
    for pkg in pip_packages:
        print(f"    - {pkg}")
PY

echo "Done: environment.yml"