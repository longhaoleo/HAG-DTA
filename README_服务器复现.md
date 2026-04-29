# HAG-DTA 服务器复现指南

---

## 1. 环境配置

### 1.1 基础环境要求
- Python 3.8
- CUDA 可用（建议）
- Linux x86_64

### 1.2 安装依赖

```bash
cd ~/HAG-DTA

# 1. 安装 PyTorch 2.0.0 GPU 版（必须根据服务器 CUDA 版本选择，以下为 CUDA 11.8 示例）
# 如果服务器是 CUDA 11.7，将 cu118 改为 cu117
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 PyTorch Geometric（匹配 PyTorch 和 CUDA 版本）
pip install torch_geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3. 安装 PyG 扩展（项目目录下已提供 Linux GPU wheel，直接安装即可）
pip install 服务器配置/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install 服务器配置/torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
pip install 服务器配置/torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
pip install 服务器配置/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl

# 4. 其他依赖
pip install rdkit-pypi pandas numpy scikit-learn scipy networkx
```

### 1.3 验证环境

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch_geometric; print(torch_geometric.__version__)"
python3 -c "from rdkit import Chem; print('rdkit ok')"
```

---

## 2. 数据预处理

进入代码目录：

```bash
cd ~/HAG-DTA/code_leo
```

### 2.1 Davis / KIBA（回归任务）

```bash
python create_data_davis_kiba.py
```

生成文件：`data/processed/davis_train.pt`、`data/processed/davis_test.pt`、`data/processed/kiba_train.pt`、`data/processed/kiba_test.pt`

### 2.2 Human / C.elegans（二分类任务）

```bash
python create_data_Human_Celegans.py
```

生成文件：`data/processed/Human_train.pt`、`Human_val.pt`、`Human_test.pt`、`Celegans_train.pt`、`Celegans_val.pt`、`Celegans_test.pt`

> ⚠️ **重要提醒**：`create_data_Human_Celegans.py` 已修改划分逻辑为真正的 8:1:1。如果之前已生成过 .pt 文件，请先删除旧文件，重新运行以生成新的划分：
> ```bash
> rm -f data/processed/Human_*.pt data/processed/Celegans_*.pt
> python create_data_Human_Celegans.py
> ```

---

## 3. 训练

训练脚本格式：

```bash
python <training_script.py> <dataset_id> <model_id>
```

- `dataset_id`: 0 或 1（见下表）
- `model_id`: 0=GIN, 1=GCN, 2=GAT, 3=SAGE

### 3.1 回归任务

| 命令 | 数据集 | 模型 |
|--------|--------|------|
| `python training_davis_kiba.py 0 0` | Davis | GIN |
| `python training_davis_kiba.py 0 1` | Davis | GCN |
| `python training_davis_kiba.py 0 2` | Davis | GAT |
| `python training_davis_kiba.py 0 3` | Davis | SAGE |
| `python training_davis_kiba.py 1 0` | KIBA | GIN |
| `python training_davis_kiba.py 1 1` | KIBA | GCN |
| `python training_davis_kiba.py 1 2` | KIBA | GAT |
| `python training_davis_kiba.py 1 3` | KIBA | SAGE |

评估指标：MSE、CI、rm²

### 3.2 二分类任务

| 命令 | 数据集 | 模型 |
|--------|--------|------|
| `python training_Human_Celegans.py 0 0` | Human | GIN |
| `python training_Human_Celegans.py 0 1` | Human | GCN |
| `python training_Human_Celegans.py 0 2` | Human | GAT |
| `python training_Human_Celegans.py 0 3` | Human | SAGE |
| `python training_Human_Celegans.py 1 0` | C.elegans | GIN |
| `python training_Human_Celegans.py 1 1` | C.elegans | GCN |
| `python training_Human_Celegans.py 1 2` | C.elegans | GAT |
| `python training_Human_Celegans.py 1 3` | C.elegans | SAGE |

评估指标：AUROC、AUPRC、Precision、Recall

---

## 4. 待修复项目前的注意事项

以下是返修前已确认的代码问题，上服务器请按此复现：

| 项 | 状态 | 说明 |
|---|------|------|
| Human/C.elegans 8:1:1 划分 | 已修 | 已修改 `create_data_Human_Celegans.py`，请重新生成 .pt |
| Human/C.elegans AUPRC bug | 已修 | 已修改 `training_Human_Celegans.py`，测试集单独重新计算 PR 曲线 |
| 随机种子 3→5 | 待修 | 需改 `training_davis_kiba.py` 和 `training_Human_Celegans.py` 中的种子列表为 `[100, 1000, 2000, 3000, 4000]`，然后重新跑实验 |
| Davis/KIBA validation 设置 | 待修 | 当前用 test set 选最优模型，存在数据泄漏风险，后续需要修改训练逻辑 |

---

## 5. 常见问题

**Q: 为什么本地 macOS 装不了？**
A: `服务器配置/` 下的 .whl 文件是 Linux x86_64 编译的，macOS 无法使用。需要在 Linux 服务器上运行。

**Q: 训练大约需要多久？**
A: 取决于 GPU 型号和设置的 NUM_EPOCHS（1000 epochs）。单个种子、单个数据集、单个模型大约几小时到十几小时不等。

**Q: 如何后台运行？**
A: 建议使用 `nohup` 或 `screen`/`tmux`：
```bash
nohup python training_davis_kiba.py 0 0 > davis_gin.log 2>&1 &
```
