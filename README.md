# HAG-DTA 服务器复现指南

Linux + CUDA GPU 环境。

---

## 环境

```bash
conda env create -f environment.yml
conda activate hag-dta

# PyG 扩展（需从 PyG 官方源安装）
pip install torch_geometric==2.6.1 torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

验证：
```bash
python -c "import torch; print(torch.cuda.is_available())"  # 必须 True
```

---

## 路径（可选覆盖）

| 变量 | 默认值 |
|------|--------|
| `HAG_DTA_CACHE_ROOT` | `/root/autodl-tmp/HAG-DTA-cache` |
| `HAG_DTA_OUTPUT_ROOT` | `/root/autodl-tmp/HAG-DTA-runs` |

---

## 数据预处理

```bash
cd ~/HAG-DTA/code_leo
python create_data_davis_kiba.py
python create_data_Human_Celegans.py
```

---

## 训练

格式：`python <script> <dataset> <model> <fold>`

| 数据集 | script | dataset_id |
|--------|--------|-----------|
| Davis | `training_davis_kiba.py` | 0 |
| KIBA | `training_davis_kiba.py` | 1 |
| Human | `training_Human_Celegans.py` | 0 |
| C.elegans | `training_Human_Celegans.py` | 1 |

model_id: 0=GIN 1=GCN 2=GAT 3=SAGE

单条示例：
```bash
python training_davis_kiba.py 0 0 0    # Davis + GIN + fold 0
```

批量（GIN 全 5 折）：
```bash
cd ~/HAG-DTA/code_leo
bash scripts/run_davis.sh &
bash scripts/run_kiba.sh &
bash scripts/run_human.sh &
bash scripts/run_celegans.sh &
```

GCN/GAT/SAGE 改脚本内 `MODEL_ID` 后同样执行。

---

## 输出

默认目录 `/root/autodl-tmp/HAG-DTA-runs/`：
- `{dataset}_Diff_DTA_GIN_fold{N}_random.csv` — 每折 5 种子最终指标
- `{dataset}训练损失_fold{N}_{seed}_{model}.csv` — 训练曲线
- `logs/{dataset}.log` — 运行日志

汇总：`python scripts/aggregate_results.py`

---

## 补充实验

| 实验 | 脚本 |
|------|------|
| Davis MMD β 消融 | `bash scripts/sensitivity_mmd_davis.sh` |
| Human MMD β 消融 | `bash scripts/sensitivity_mmd_human.sh` |
| n1/n2 敏感性分析 | `bash scripts/sensitivity_n1n2_all.sh` |
