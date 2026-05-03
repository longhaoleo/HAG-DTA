# 实验运行计划


## 一、主力实验（GIN 模型）

每个脚本运行一个数据集的 GIN 实验（MODEL_ID=0），包含 5 个 fold，每个 fold 内部自动跑 5 个随机种子。

| 脚本 | 数据集 | 任务 | Fold | 种子 | 实验数 |
|------|--------|------|------|------|--------|
| `run_davis.sh` | Davis | 回归 | 5 | 5 | 25 |
| `run_kiba.sh` | KIBA | 回归 | 5 | 5 | 25 |
| `run_human.sh` | Human | 分类 | 5 | 5 | 25 |
| `run_celegans.sh` | C.elegans | 分类 | 5 | 5 | 25 |
| **小计** | | | | | **100** |

每个脚本的输出文件：
- `OUTPUT/{dataset}_Diff_DTA_GIN_fold{N}_random.csv`（每个 fold 一份，含 5 个种子的指标）
- `OUTPUT/{dataset}训练损失_fold{N}_{seed}_{model_name}.csv`
- 日志：`OUTPUT/logs/{dataset}.log`

---

## 二、GCN / GAT / SAGE 模型

脚本只支持 GIN（MODEL_ID=0）。运行其他模型需要修改脚本中的 MODEL_ID：

| 模型 | MODEL_ID | 各数据集实验数 | 总实验数 |
|------|----------|-------------|---------|
| GCN | 1 | 25 × 4 | 100 |
| GAT | 2 | 25 × 4 | 100 |
| SAGE | 3 | 25 × 4 | 100 |
| **小计** | | | **300** |

执行方式（以 Davis + GCN 为例）：

```bash
# 把脚本中 MODEL_ID=0 改为 MODEL_ID=1，然后运行
sed -i 's/MODEL_ID=0/MODEL_ID=1/' scripts/run_davis.sh
bash scripts/run_davis.sh
```

或者复制脚本并改参数：

```bash
for mid in 1 2 3; do
    for script in run_davis.sh run_kiba.sh run_human.sh run_celegans.sh; do
        cp scripts/$script scripts/${script%.sh}_model${mid}.sh
        sed -i '' "s/MODEL_ID=0/MODEL_ID=$mid/" scripts/${script%.sh}_model${mid}.sh
    done
done
```

---

## 三、敏感性分析

每个敏感性脚本只跑 1 个 fold + 1 个种子，用于快速评估参数影响。

### 3.1 n1 / n2 敏感度（回应审稿人超参数质疑）

| 脚本 | 数据集 | 参数组合数 | 实验数 |
|------|--------|----------|--------|
| `sensitivity_n1n2.sh` | Davis | 11 | 11 |
| `sensitivity_n1n2_human.sh` | Human | 11 | 11 |
| **小计** | | | **22** |

11 组 (n1, n2)：`(4,2) (5,2) (5,3) (6,2) (6,3) (6,4) (7,2) (7,3) (7,4) (8,3) (8,4)`

其中 (6,3) 为当前默认值。

### 3.2 MMD 系数消融（回应审稿人对 β=0.05 的质疑）

| 脚本 | 数据集 | 参数组合数 | 实验数 |
|------|--------|----------|--------|
| `sensitivity_mmd.sh` | Davis | 5 | 5 |

5 组 β：`0, 0.01, 0.05, 0.1, 0.5`

---

## 四、总实验量

| 类别 | 实验数 | 备注 |
|------|--------|------|
| GIN 主力 | 100 | 4 数据集 × 5 fold × 5 种子 |
| GCN | 100 | 需改 MODEL_ID |
| GAT | 100 | 需改 MODEL_ID |
| SAGE | 100 | 需改 MODEL_ID |
| n1/n2 敏感性 | 22 | 2 数据集 × 11 组, 1 fold × 1 种子 |
| MMD 敏感性 | 5 | 1 数据集 × 5 组, 1 fold × 1 种子 |
| **总计** | **427** | |

---

## 五、执行顺序

先跑敏感性分析（快，验证脚本正确），最后跑完整的 k-fold 主力实验。

```
Step 1 ─ 敏感性分析（消融实验，1 fold × 1 seed，快速验证）
    ├── sensitivity_mmd.sh              # MMD beta 系数（Davis，5 组）
    ├── sensitivity_n1n2.sh             # n1/n2 敏感度（Davis，11 组）
    └── sensitivity_n1n2_human.sh       # n1/n2 敏感度（Human，11 组）

Step 2 ─ 确认敏感性正常后，并行跑 GIN 主力实验（5 fold × 5 seed）
    ├── run_davis.sh    &
    ├── run_kiba.sh     &
    ├── run_human.sh    &
    └── run_celegans.sh &

Step 3 ─ 确认 GIN 无误后，跑 GCN / GAT / SAGE（改 MODEL_ID 后执行）
```

---

## 六、注意事项

- 训练脚本内部已配置 5 个种子：`SEEDS = [100, 1000, 2000, 3000, 4000]`（见 `config/training.py`）
- 输出目录默认为 `/root/autodl-tmp/HAG-DTA-runs`，可通过 `HAG_DTA_OUTPUT_ROOT` 环境变量修改
- 代码数据划分为 64% train / 16% val / 20% test（论文需同步修改为这一口径）
- MMD 系数消融中的 β=0.05 为当前默认值
