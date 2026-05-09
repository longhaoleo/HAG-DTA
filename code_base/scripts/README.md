# 实验运行计划（code_base）

## 当前实验协议

- 工作目录：`~/HAG-DTA/code_base`
- 回归任务（Davis / KIBA）：对齐 baseline，使用 DeepDTA 原始 classic split
- 分类任务（Human / C.elegans）：对齐 Tsubaki / TransformerCPI 路线，使用单次随机 `80/10/10` 划分
- 分类任务划分随机种子：`1234`
- 所有主实验默认 5 个随机种子：`100, 1000, 2000, 3000, 4000`
- 输出目录默认：`/root/autodl-tmp/HAG-DTA-runs`

## 一、主实验脚本

### 1. 回归：Davis / KIBA

每个 `run_*.sh` 都是单 seed 脚本。默认只跑 HAG-DTA GIN；第二个参数可选择你自己的其它 HAG-DTA 变体或 `all`。

| 脚本 | 数据集 | 模式 | 输入 | 实际运行 |
|------|--------|------|------|----------|
| `run_davis.sh` | Davis | classic split | `<seed> [models]` | 默认 HAG-DTA GIN |
| `run_kiba.sh` | KIBA | classic split | `<seed> [models]` | 默认 HAG-DTA GIN |
| `run_davis_full.sh` | Davis | classic split | `[models]` | 默认 5 seeds × HAG-DTA GIN |
| `run_kiba_full.sh` | KIBA | classic split | `[models]` | 默认 5 seeds × HAG-DTA GIN |

HAG-DTA 回归 `models` 可选：`gin`、`gcn`、`gat`、`sage`、`all`。

示例：

```bash
cd ~/HAG-DTA/code_base
bash scripts/run_davis.sh 100
bash scripts/run_kiba.sh 100
bash scripts/run_davis.sh 100 gat
bash scripts/run_kiba_full.sh all
```

### 2. 分类：Human / C.elegans

分类任务现在不是 k-fold，而是单次 `80/10/10` 随机划分。单 seed 脚本默认只跑 HAG-DTA GIN；第二个参数可选择其它 HAG-DTA 变体或 `all`。

| 脚本 | 数据集 | 模式 | 输入 | 实际运行 |
|------|--------|------|------|----------|
| `run_human.sh` | Human | single random split (80/10/10, seed=1234) | `<seed> [models]` | 默认 HAG-DTA GIN |
| `run_celegans.sh` | C.elegans | single random split (80/10/10, seed=1234) | `<seed> [models]` | 默认 HAG-DTA GIN |
| `run_human_full.sh` | Human | single random split (80/10/10, seed=1234) | `[models]` | 默认 5 seeds × HAG-DTA GIN |
| `run_celegans_full.sh` | C.elegans | single random split (80/10/10, seed=1234) | `[models]` | 默认 5 seeds × HAG-DTA GIN |

分类 `models` 可选：`gin`、`gcn`、`gat`、`sage`、`all`。

示例：

```bash
cd ~/HAG-DTA/code_base
python create_data_Human_Celegans.py
bash scripts/run_human.sh 100
bash scripts/run_celegans.sh 100
bash scripts/run_human_full.sh all
```

## 二、敏感性分析与基线

### 1. n1 / n2 网格

| 脚本 | 数据集 | 模式 | 组合数 | 说明 |
|------|--------|------|--------|------|
| `sensitivity_n1n2_davis.sh` | Davis | classic split | 14 | 单 seed，默认 `SEED=100` |
| `sensitivity_n1n2_kiba.sh` | KIBA | classic split | 14 | 单 seed，默认 `SEED=100` |
| `sensitivity_n1n2_human.sh` | Human | 80/10/10 single split | 14 | 单 seed，默认 `SEED=100` |
| `sensitivity_n1n2_celegans.sh` | C.elegans | 80/10/10 single split | 14 | 单 seed，默认 `SEED=100` |
| `sensitivity_n1n2_all.sh` | 全部四个数据集 | 各自 split | 每个数据集 14 | 单 seed，默认 `SEED=100` |

示例：

```bash
bash scripts/sensitivity_n1n2_all.sh
SEED=1000 bash scripts/sensitivity_n1n2_all.sh
```

### 2. MMD β 消融

| 脚本 | 数据集 | 模式 | β |
|------|--------|------|---|
| `sensitivity_mmd.sh` | Davis | classic split，单 seed，默认 `SEED=100` | `0, 0.01, 0.05, 0.1, 0.5, 1.0` |

### 3. GraphDTA 基线

GraphDTA baseline 单独运行，不混在 HAG-DTA 主脚本里。

| 脚本 | 数据集 | 模式 | 输入 | 实际运行 |
|------|--------|------|------|----------|
| `run_davis_graphdta.sh` | Davis | classic split | `<seed> [models]` | 默认 GraphDTA GIN |
| `run_kiba_graphdta.sh` | KIBA | classic split | `<seed> [models]` | 默认 GraphDTA GIN |
| `run_davis_graphdta_full.sh` | Davis | classic split | `[models]` | 默认 5 seeds × GraphDTA GIN |
| `run_kiba_graphdta_full.sh` | KIBA | classic split | `[models]` | 默认 5 seeds × GraphDTA GIN |

GraphDTA `models` 可选：`gin`、`gcn`、`gat`、`sage`、`all`。

示例：

```bash
cd ~/HAG-DTA/code_base
bash scripts/run_davis_graphdta.sh 100
bash scripts/run_kiba_graphdta_full.sh all
```

### 4. TransformerCPI 基线

新增文件：

- `create_data_transformercpi.py`：按当前 Human / C.elegans 的 `80/10/10` 划分构造 TransformerCPI 输入
- `model_transformercpi.py`：官方 TransformerCPI 结构重写版
- `training_transformercpi.py`：官方训练逻辑适配版（5 seeds）
- `scripts/run_human_transformercpi.sh`
- `scripts/run_celegans_transformercpi.sh`
- `scripts/statistical_tests_local.py`：本地 HAG-DTA vs TransformerCPI 显著性比较

运行顺序：

```bash
cd ~/HAG-DTA/code_base
python create_data_transformercpi.py
bash scripts/run_human_transformercpi.sh
bash scripts/run_celegans_transformercpi.sh
python scripts/statistical_tests_local.py
```

补充说明：

- `code_base/resources/transformercpi/word2vec_30.model` 已复制到本地
- 预处理需要 `gensim`、`rdkit`、`torch`
- 当前实现为了和你现有 HAG-DTA 分类实验公平比较，使用相同的 Human / C.elegans 原始 txt 与同一 `80/10/10` 划分；没有沿用官方脚本里“过滤带 `.` 的 SMILES”那一步（原始数据中这类样本数量为：Human `516` 条，C.elegans `275` 条；若过滤会改变比较样本集）

## 三、输出文件约定

### 回归（classic）

- 主结果：`{dataset}_{model}_random.csv`
- 训练曲线：`{dataset}训练损失_{seed}_{model}.csv`
- 注意力：`{dataset}注意力分数_{seed}_{model}.csv`

### 分类（single split）

- 主结果：`{dataset}_{model}_random.csv`
- 训练曲线：`{dataset}训练损失_{seed}_{model}.csv`

## 四、建议执行顺序

1. 先重建分类数据
2. 先跑敏感性 / 小规模验证
3. 再跑分类主实验
4. 最后跑 KIBA 全量

推荐顺序：

```bash
cd ~/HAG-DTA/code_base

# 1) 分类数据重建（Human / C.elegans 使用 80/10/10, seed=1234）
python create_data_Human_Celegans.py

# 2) 快速验证
bash scripts/sensitivity_n1n2_davis.sh
bash scripts/sensitivity_n1n2_all.sh
bash scripts/sensitivity_mmd.sh

# 3) 分类主实验
bash scripts/run_human_full.sh
bash scripts/run_celegans_full.sh

# 4) 回归主实验（示例：分 seed 跑）
bash scripts/run_davis.sh 100
bash scripts/run_kiba.sh 100
```

## 五、注意事项

- 分类任务的旧 fold 结果不要再和当前结果混用
- 重新划分后，必须重新运行 `create_data_Human_Celegans.py`
- `scripts/statistical_tests.py` 已同时兼容：
  - 回归 `*_random.csv`
  - 分类 `*_random.csv`
