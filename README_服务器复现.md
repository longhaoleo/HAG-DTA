# HAG-DTA 服务器复现指南

本文件为 Linux + NVIDIA GPU 服务器环境下的完整复现流程。本地 macOS 无法直接运行，因为 PyG 扩展需 Linux x86_64，且需要 CUDA。

---

## 1. 环境配置

### 1.1 基础环境要求

- Python 3.8
- CUDA 可用（建议 CUDA 11.8）
- Linux x86_64

### 1.2 安装依赖

```bash
cd ~/HAG-DTA

# 1. 安装 PyTorch 2.0.0（根据服务器 CUDA 版本选择，cu118 / cu117 / cu121）
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 PyTorch Geometric
pip install torch_geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3. 安装 PyG 扩展
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 如果 PyG 官方源下载慢或网络不稳定，可使用项目内的本地 wheel 备选：
# pip install 服务器配置/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
# pip install 服务器配置/torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
# pip install 服务器配置/torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
# pip install 服务器配置/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl

# 4. 其他依赖
pip install pandas numpy scipy scikit-learn networkx rdkit-pypi matplotlib pillow
```

### 1.3 验证环境

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "from rdkit import Chem; print('rdkit ok')"
```

`torch.cuda.is_available()` 返回 `False` 的话训练会非常慢，检查 CUDA 安装。

---

## 2. 路径配置（可选）

代码通过 `code_leo/config/paths.py` 统一管理路径，支持环境变量覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HAG_DTA_DATA_ROOT` | `data` | 原始数据目录 |
| `HAG_DTA_CACHE_ROOT` | `/root/autodl-tmp/HAG-DTA-cache` | 预处理 `.pt` 和 fold 索引 |
| `HAG_DTA_OUTPUT_ROOT` | `/root/autodl-tmp/HAG-DTA-runs` | 训练输出（模型 + CSV） |

如果服务器数据盘路径不同，建议：

```bash
export HAG_DTA_CACHE_ROOT=/your/data/drive/HAG-DTA-cache
export HAG_DTA_OUTPUT_ROOT=/your/data/drive/HAG-DTA-runs
```

或直接修改 `code_leo/config/paths.py`。

---

## 3. 数据预处理

进入代码目录：

```bash
cd ~/HAG-DTA/code_leo
```

### 3.1 Davis / KIBA（回归任务）

```bash
python create_data_davis_kiba.py
```

生成文件：

| 文件 | 说明 |
|------|------|
| `<CACHE_ROOT>/processed/davis_all.pt` | Davis 全量图数据 |
| `<CACHE_ROOT>/processed/kiba_all.pt` | KIBA 全量图数据（小分子，≤46 原子） |
| `<CACHE_ROOT>/processed/kiba_all1.pt` | KIBA 大分子分支（>46 原子） |
| `<CACHE_ROOT>/fold_indices/davis_fold0.json` ~ `davis_fold4.json` | Davis 5 折索引 |
| `<CACHE_ROOT>/fold_indices/kiba_fold0.json` ~ `kiba_fold4.json` | KIBA 5 折索引 |

划分方式：先固定 20% 作为 test，剩余 80% 做 5-fold（每 fold: train 64% / val 16% / test 20%）。

### 3.2 Human / C.elegans（二分类任务）

```bash
python create_data_Human_Celegans.py
```

生成文件：

| 文件 | 说明 |
|------|------|
| `<CACHE_ROOT>/processed/Human_all.pt` | Human 全量图数据 |
| `<CACHE_ROOT>/processed/Celegans_all.pt` | C.elegans 全量图数据 |
| `<CACHE_ROOT>/fold_indices/Human_fold0.json` ~ `Human_fold4.json` | Human 5 折索引 |
| `<CACHE_ROOT>/fold_indices/Celegans_fold0.json` ~ `Celegans_fold4.json` | C.elegans 5 折索引 |

### 3.3 预处理注意事项

> ⚠️ 如果之前已生成过 `.pt` 或 fold 索引，修改划分逻辑后必须先删除旧文件再重新运行：

```bash
rm -f /root/autodl-tmp/HAG-DTA-cache/processed/Human_*.pt
rm -f /root/autodl-tmp/HAG-DTA-cache/processed/Celegans_*.pt
rm -f /root/autodl-tmp/HAG-DTA-cache/processed/davis_*.pt
rm -f /root/autodl-tmp/HAG-DTA-cache/processed/kiba_*.pt
rm -f /root/autodl-tmp/HAG-DTA-cache/fold_indices/*.json
python create_data_davis_kiba.py
python create_data_Human_Celegans.py
```

---

## 4. 训练

### 4.1 命令格式

```bash
python <训练脚本> <dataset_id> <model_id> <fold_id>
```

参数说明：

| 参数 | 取值 | 含义 |
|------|------|------|
| `dataset_id` | 见下表 | 数据集选择 |
| `model_id` | 0=GIN, 1=GCN, 2=GAT, 3=SAGE | 图卷积算子 |
| `fold_id` | 0~4 | 5 折交叉验证的第几折 |

### 4.2 回归任务（Davis / KIBA）

进入代码目录：`cd ~/HAG-DTA/code_leo`

| 命令 | 数据集 | 模型 |
|------|--------|------|
| `python training_davis_kiba.py 0 0 0` | Davis | GIN |
| `python training_davis_kiba.py 0 1 0` | Davis | GCN |
| `python training_davis_kiba.py 0 2 0` | Davis | GAT |
| `python training_davis_kiba.py 0 3 0` | Davis | SAGE |
| `python training_davis_kiba.py 1 0 0` | KIBA | GIN |
| `python training_davis_kiba.py 1 1 0` | KIBA | GCN |
| `python training_davis_kiba.py 1 2 0` | KIBA | GAT |
| `python training_davis_kiba.py 1 3 0` | KIBA | SAGE |

以上是 fold 0 的示例，完整 5 折需 fold 0~4 各跑一遍。

评估指标：MSE、CI、rm²。使用 **validation set** 选最优模型（`VAL_INTERVAL=5` epochs 验证一次），test set 仅最终评估。支持 early stopping（`EARLY_STOP_PATIENCE=50`）。

### 4.3 二分类任务（Human / C.elegans）

进入代码目录：`cd ~/HAG-DTA/code_leo`

| 命令 | 数据集 | 模型 |
|------|--------|------|
| `python training_Human_Celegans.py 0 0 0` | Human | GIN |
| `python training_Human_Celegans.py 0 1 0` | Human | GCN |
| `python training_Human_Celegans.py 0 2 0` | Human | GAT |
| `python training_Human_Celegans.py 0 3 0` | Human | SAGE |
| `python training_Human_Celegans.py 1 0 0` | C.elegans | GIN |
| `python training_Human_Celegans.py 1 1 0` | C.elegans | GCN |
| `python training_Human_Celegans.py 1 2 0` | C.elegans | GAT |
| `python training_Human_Celegans.py 1 3 0` | C.elegans | SAGE |

评估指标：AUROC、AUPRC、Precision、Recall。每个 epoch 在 validation set 上评估，用最高 AUROC 保存模型，test set 在最优时间点重新计算指标（AUPRC 单独重算，已修复）。

### 4.4 跑完整 5 折

```bash
cd ~/HAG-DTA/code_leo

# Davis + GIN, 全部 5 折
for fold in 0 1 2 3 4; do
    python training_davis_kiba.py 0 0 ${fold}
done

# Human + GIN, 全部 5 折
for fold in 0 1 2 3 4; do
    python training_Human_Celegans.py 0 0 ${fold}
done
```

---

## 5. 输出文件

训练产物默认保存到 `HAG_DTA_OUTPUT_ROOT`（默认 `/root/autodl-tmp/HAG-DTA-runs`）。

### 5.1 目录结构

```
/root/autodl-tmp/HAG-DTA-runs/
├── checkpoints/
│   ├── model_davis_Diff_DTA_GIN_fold0_100.model
│   ├── model_davis_Diff_DTA_GIN_fold0_1000.model
│   └── ...
├── davis训练损失_fold0_100_Diff_DTA_GIN.csv
├── davis注意力分数_fold0_100_Diff_DTA_GIN.csv
├── davis_Diff_DTA_GIN_fold0_random.csv
└── ...
```

其中 `*_random.csv` 是各种子最终 test 指标汇总。

### 5.2 默认超参数

| 参数 | 回归 | 分类 |
|------|------|------|
| Epochs | 1000 | 1000 |
| Train batch size | 512 | 256 |
| Test batch size | 512 | 2048 |
| Learning rate | 0.0005 | 0.0005 |
| Seeds | 100, 1000, 2000 | 100, 1000, 2000 |
| Val interval | 5 epochs | 每 epoch |
| Early stop | 50 epochs | 无 |

超参数在 `code_leo/config/training.py` 中统一配置。

---

## 6. 后台运行

建议用 `nohup` 或 `tmux`：

```bash
cd ~/HAG-DTA/code_leo
nohup python training_davis_kiba.py 0 0 0 > /root/autodl-tmp/HAG-DTA-runs/davis_gin_fold0.log 2>&1 &
```

查看日志：

```bash
tail -f /root/autodl-tmp/HAG-DTA-runs/davis_gin_fold0.log
```

批量后台跑（以 Davis 全模型 fold 0 为例）：

```bash
cd ~/HAG-DTA/code_leo
for m in 0 1 2 3; do
    nohup python training_davis_kiba.py 0 ${m} 0 > /root/autodl-tmp/HAG-DTA-runs/davis_model${m}_fold0.log 2>&1 &
done
```

---

## 7. 当前待修复项

返修前已确认的代码问题状态：

| 项 | 状态 | 说明 |
|---|------|------|
| 数据划分修复（Human/C.elegans） | ✅ 已修 | 已从 64/16/20 改为 80/10/10，需重新生成 `.pt` |
| AUPRC 计算 bug | ✅ 已修 | 测试集单独重新计算 precision_recall_curve |
| Davis/KIBA 数据泄漏 | ✅ 已修 | 已改为 validation 选模型 + test 最终评估 |
| 5-fold 交叉验证 | ✅ 已修 | 创建数据时生成 fold 索引，训练支持 fold_id 参数 |
| 随机种子 3→5 | ⚠️ 待修 | 需改 `config/training.py` 中 `SEEDS = [100, 1000, 2000, 3000, 4000]`，然后重新跑全部实验 |
| 统计显著性检验 | ⚠️ 待补 | 审稿人要求 t-test / ANOVA |

---

## 8. 运行前检查清单

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `<CACHE_ROOT>/processed/` 下有 `*_all.pt` 文件
- [ ] `<CACHE_ROOT>/fold_indices/` 下有 `*_fold*.json` 文件
- [ ] `HAG_DTA_OUTPUT_ROOT` 目录可写
- [ ] 当前代码是 `code_leo/` 目录（非旧版 `code/`）
- [ ] （可选）如果需要 5 个种子，已改 `config/training.py` 的 `SEEDS`

---

## 9. 常见问题

**Q: 为什么本地 macOS 装不了？**
A: PyG 扩展的 wheel 是 Linux x86_64 + CUDA 编译的，macOS 无法使用。需要在 Linux GPU 服务器上运行。

**Q: 提示"please run create_data_*.py to prepare"？**
A: `CACHE_ROOT` 下缺少 `.pt` 或 fold 索引，先运行预处理脚本。

**Q: 训练大约需要多久？**
A: 单种子、单 fold、单数据集、单模型大约几小时到十几小时（取决于 GPU，默认 1000 epochs + early stopping 通常提前停止）。

**Q: KIBA 为什么有 `_all1.pt`？**
A: KIBA 中等大分子（>46 原子）被单独用更大的 `ToDense` 节点上限处理，训练时会同时加载两个 DataLoader。

**Q: 改了划分后为什么结果和以前不一样？**
A: `.pt` 是预处理产物。改脚本后必须删除旧文件重新生成，否则沿用旧数据。

**Q: 如何知道训练跑了几个种子？**
A: 看 `config/training.py` 的 `SEEDS` 列表，或输出目录下的 checkpoint 文件命名（如 `_100.model` 即 seed=100）。

---

## 10. 最快复现路径

```bash
cd ~/HAG-DTA/code_leo

# 预处理
python create_data_davis_kiba.py
python create_data_Human_Celegans.py

# 回归：Davis + GIN, fold 0
python training_davis_kiba.py 0 0 0

# 分类：Human + GIN, fold 0
python training_Human_Celegans.py 0 0 0
```

确认一条能跑通后再批量跑完整实验。
