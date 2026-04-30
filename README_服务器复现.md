# HAG-DTA 服务器复现 README

这份文档是当前项目的服务器运行说明，重点回答 4 个问题：

1. 环境怎么配
2. 数据怎么预处理
3. 训练命令怎么跑
4. 结果会保存到哪里

默认面向 Linux + NVIDIA GPU 服务器。

---

## 1. 当前代码口径

当前 `code_leo/` 中的代码采用以下实验口径：

- 四个数据集都使用 `all.pt + fold 索引` 的方式组织数据
- 先固定 `20%` 作为 `test`
- 再对剩余 `80%` 做 `5-fold`
- 单次训练对应的有效比例约为 `64% / 16% / 20%`
  - `train / validation / test`
- 回归任务 `Davis / KIBA`
  - 使用 `validation set` 选择最优模型
  - `test set` 仅用于最终评估
- 分类任务 `Human / C.elegans`
  - 同样使用 `validation set` 选择最优模型
- 路径配置统一放在：

```python
code_leo/config/paths.py
```

- 训练超参数统一放在：

```python
code_leo/config/training.py
```

当前主要可控项包括：

- `DATA_ROOT`
- `CACHE_ROOT`
- `PROCESSED_DIR`
- `FOLD_DIR`
- `OUTPUT_ROOT`
- `CHECKPOINT_DIR`

训练侧常用可控项包括：

- `SEEDS`
- `CUDA_NAME`
- `REGRESSION_TRAINING`
- `CLASSIFICATION_TRAINING`

如果你的服务器数据盘路径不同，优先改：

- [paths.py](/Users/leo/HAG-DTA/code_leo/config/paths.py:1)

也可以直接用环境变量覆盖：

```bash
export HAG_DTA_DATA_ROOT=/path/to/data
export HAG_DTA_CACHE_ROOT=/root/autodl-tmp/HAG-DTA-cache
export HAG_DTA_OUTPUT_ROOT=/root/autodl-tmp/HAG-DTA-runs
```

---

## 2. 目录说明

项目里当前主要用这几个目录：

- `code_leo/`
  - 当前实验代码
- `code_leo/config/`
  - 路径配置
- `data/`
  - 原始数据
- `/root/autodl-tmp/HAG-DTA-cache`
  - 预处理 `.pt` 和 fold 索引
- `服务器配置/`
  - 旧版 Linux wheel 备份
- `/root/autodl-tmp/HAG-DTA-runs`
  - 训练输出目录

建议服务器上这样进入项目：

```bash
cd ~/HAG-DTA
```

---

## 3. 环境配置

### 3.1 推荐基础环境

- Python `3.8`
- PyTorch `2.0.0`
- CUDA `11.8`

如果你在 AutoDL 或类似环境上跑，建议单独建一个环境。

### 3.2 安装命令

```bash
cd ~/HAG-DTA

# 1. PyTorch
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 2. PyG
pip install torch_geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3. 其他依赖
pip install pandas numpy scipy scikit-learn networkx rdkit-pypi matplotlib pillow
```

如果官方源安装失败，可以尝试项目里已有的 wheel，但那几份 wheel 比较旧，不优先推荐。

### 3.3 环境检查

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch_geometric; print(torch_geometric.__version__)"
python3 -c "from rdkit import Chem; print('rdkit ok')"
```

如果 `torch.cuda.is_available()` 返回 `False`，训练会非常慢。

---

## 4. 数据预处理

进入代码目录：

```bash
cd ~/HAG-DTA/code_leo
```

### 4.1 Davis / KIBA

```bash
python3 create_data_davis_kiba.py
```

当前会生成：

- `data/processed/davis_all.pt`
- `data/processed/kiba_all.pt`
- `data/processed/kiba_all1.pt`
- `data/fold_indices/davis_fold0.json` 到 `davis_fold4.json`
- `data/fold_indices/kiba_fold0.json` 到 `kiba_fold4.json`

默认实际位置为：

- `/root/autodl-tmp/HAG-DTA-cache/processed/*.pt`
- `/root/autodl-tmp/HAG-DTA-cache/fold_indices/*.json`

说明：

- `*_all.pt` 只缓存一次全量图数据
- `kiba_all1.pt` 是大分子分支
- 每个 fold 只保存索引，不重复保存 5 套 `.pt`

### 4.2 Human / C.elegans

```bash
python3 create_data_Human_Celegans.py
```

当前会生成：

- `data/processed/Human_all.pt`
- `data/processed/Celegans_all.pt`
- `data/fold_indices/Human_fold0.json` 到 `Human_fold4.json`
- `data/fold_indices/Celegans_fold0.json` 到 `Celegans_fold4.json`

### 4.3 预处理注意事项

- 修改了划分逻辑后，必须重新跑预处理
- 如果想强制重建 `.pt`，可以先删除 `data/processed/` 下对应文件再运行
- 如果想强制重建 fold 划分，可以同时删除 `data/fold_indices/` 下对应 json

如果使用当前默认缓存配置，上面两项实际对应的是：

- `/root/autodl-tmp/HAG-DTA-cache/processed/`
- `/root/autodl-tmp/HAG-DTA-cache/fold_indices/`

---

## 5. 训练命令

训练前先确认路径配置正确。当前脚本会自动创建输出目录和 checkpoint 目录。

默认输出根目录是：

```bash
/root/autodl-tmp/HAG-DTA-runs
```

### 5.1 参数说明

脚本格式：

```bash
python3 <script> <dataset_id> <model_id> <fold_id>
```

其中：

- `dataset_id`
  - 回归任务：`0=davis`, `1=kiba`
  - 分类任务：`0=Human`, `1=Celegans`
- `model_id`
  - `0=GIN`
  - `1=GCN`
  - `2=GAT`
  - `3=SAGE`
- `fold_id`
  - `0~4`
  - 表示当前使用哪个 validation fold

### 5.2 回归任务

```bash
cd ~/HAG-DTA/code_leo

python3 training_davis_kiba.py 0 0 0   # Davis + GIN + fold0
python3 training_davis_kiba.py 0 1 0   # Davis + GCN + fold0
python3 training_davis_kiba.py 0 2 0   # Davis + GAT + fold0
python3 training_davis_kiba.py 0 3 0   # Davis + SAGE + fold0

python3 training_davis_kiba.py 1 0 0   # KIBA + GIN + fold0
python3 training_davis_kiba.py 1 1 0   # KIBA + GCN + fold0
python3 training_davis_kiba.py 1 2 0   # KIBA + GAT + fold0
python3 training_davis_kiba.py 1 3 0   # KIBA + SAGE + fold0
```

### 5.3 分类任务

```bash
cd ~/HAG-DTA/code_leo

python3 training_Human_Celegans.py 0 0 0   # Human + GIN + fold0
python3 training_Human_Celegans.py 0 1 0   # Human + GCN + fold0
python3 training_Human_Celegans.py 0 2 0   # Human + GAT + fold0
python3 training_Human_Celegans.py 0 3 0   # Human + SAGE + fold0

python3 training_Human_Celegans.py 1 0 0   # Celegans + GIN + fold0
python3 training_Human_Celegans.py 1 1 0   # Celegans + GCN + fold0
python3 training_Human_Celegans.py 1 2 0   # Celegans + GAT + fold0
python3 training_Human_Celegans.py 1 3 0   # Celegans + SAGE + fold0
```

### 5.4 跑完整 5 折

四个数据集现在都需要把 `fold_id=0~4` 跑完。

例如 Davis + GIN：

```bash
cd ~/HAG-DTA/code_leo
for fold in 0 1 2 3 4; do
  python3 training_davis_kiba.py 0 0 ${fold}
done
```

---

## 6. 输出文件位置

当前训练产物默认都保存到：

```bash
/root/autodl-tmp/HAG-DTA-runs
```

包括：

- 最优模型 checkpoint
- `训练损失_*.csv`
- `注意力分数_*.csv`
- `*_random.csv`

其中 checkpoint 默认保存在：

```bash
/root/autodl-tmp/HAG-DTA-runs/checkpoints
```

### 6.1 回归任务输出示例

- `model_davis_Diff_DTA_GIN_fold0_100.model`
- `davis训练损失_fold0_100_Diff_DTA_GIN.csv`
- `davis注意力分数_fold0_100_Diff_DTA_GIN.csv`
- `davis_Diff_DTA_GIN_fold0_random.csv`

### 6.2 分类任务输出示例

- `model_Human_Diff_DTA_GIN_fold0_100.model`
- `Human_Diff_DTA_GIN_fold0_random.csv`

---

## 7. 后台运行建议

建议使用 `nohup` 或 `tmux`。

### 7.1 nohup

```bash
cd ~/HAG-DTA/code_leo
nohup python3 training_davis_kiba.py 0 0 0 > /root/autodl-tmp/HAG-DTA-runs/davis_gin_fold0.log 2>&1 &
```

### 7.2 查看日志

```bash
tail -f /root/autodl-tmp/HAG-DTA-runs/davis_gin.log
```

---

## 8. 当前默认超参数

### 回归任务

- epoch: `1000`
- batch size: `512`
- lr: `0.0005`
- seeds: `100, 1000, 2000`

### 分类任务

- epoch: `1000`
- batch size: `256`
- test batch size: `2048`
- lr: `0.0005`
- seeds: `100, 1000, 2000`

如果后面补论文要求的 5 个随机种子，需要把两个训练脚本里的 seed 列表改成：

```python
[100, 1000, 2000, 3000, 4000]
```

---

## 9. 运行前检查清单

正式跑之前建议确认：

1. `OUTPUT_ROOT` 是否指向数据盘
2. `.pt` 是否已经按当前划分重新生成
3. 当前脚本是否是 `code_leo/` 这一版，而不是旧的 `code/`
4. GPU 是否可用
5. 日志输出路径是否可写

---

## 10. 常见问题

### Q1. 为什么跑起来提示先运行 `create_data_*.py`？

说明 `data/processed/` 下缺少当前脚本要读取的 `.pt` 文件，先重新跑预处理。

### Q2. 为什么 KIBA 会有 `train1 / val1 / test1`？

因为 KIBA 里较大的分子被单独分出去，使用不同的 `ToDense` 节点上限。

### Q3. 为什么结果文件不在代码目录里？

因为当前脚本已把所有结果统一写到：

```bash
/root/autodl-tmp/HAG-DTA-runs
```

### Q4. 改了划分后为什么结果和以前不一样？

因为 `.pt` 是预处理产物。只改脚本不重建 `.pt`，不会自动切换到新划分。

---

## 11. 推荐最小流程

如果你只是想最快复现一组结果，建议直接跑：

```bash
cd ~/HAG-DTA/code_leo

python3 create_data_davis_kiba.py
python3 training_davis_kiba.py 0 0
```

或者分类任务：

```bash
cd ~/HAG-DTA/code_leo

python3 create_data_Human_Celegans.py
python3 training_Human_Celegans.py 0 0
```

这就是当前代码库的最小复现路径。
