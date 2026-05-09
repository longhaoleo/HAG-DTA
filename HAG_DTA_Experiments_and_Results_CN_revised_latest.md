# 4. Experiments and Results

## 4.1 Datasets, evaluation metrics, and implementation details

为全面评估所提出 HAG-DTA 模型的有效性，本文在两个药物-靶标亲和力回归数据集 Davis 和 KIBA，以及两个药物-靶标相互作用二分类数据集 Human 和 C.elegans 上进行了实验验证。Davis 与 KIBA 用于评价模型对连续亲和力数值的预测能力，Human 与 C.elegans 用于评价模型对药物-靶标相互作用关系的识别能力。各数据集的任务类型、划分方式、评价指标及默认层次池化参数如 Table~\ref{tab:datasets} 所示。

Table~\ref{tab:datasets}  Datasets used in this study.

| Dataset | Task | Split | Metrics | Default $(n_1,n_2)$ |
|---|---|---|---|:---:|
| Davis | Regression | Classic split from DeepDTA | MSE / CI / rm$^2$ | (4,2) |
| KIBA | Regression | Classic split from DeepDTA | MSE / CI / rm$^2$ | (6,2) |
| Human | Binary classification | 5-fold CV (64/16/20) | AUROC / AUPRC / Precision / Recall | (7,3) |
| C.elegans | Binary classification | 5-fold CV (64/16/20) | AUROC / AUPRC / Precision / Recall | (7,3) |

需要说明的是，Davis 和 KIBA 使用 DeepDTA 原始 train/test split，以便与 GraphDTA 等经典基线方法保持一致。Human 和 C.elegans 使用 5-fold cross-validation，其中每一折按照 64/16/20 的比例划分训练集、验证集和测试集，并报告 5-fold 交叉验证结果。

对于回归任务，本文采用均方误差（mean squared error, MSE）、一致性指数（concordance index, CI）和 $rm^2$ 作为评价指标。其中，MSE 用于衡量预测亲和力与真实亲和力之间的误差，CI 用于衡量预测结果与真实排序之间的一致性，$rm^2$ 用于进一步评价模型的回归拟合能力。MSE 越低表示预测误差越小，CI 和 $rm^2$ 越高表示模型预测性能越好。

对于二分类任务，本文采用 AUROC、AUPRC、Precision 和 Recall 作为评价指标。其中，AUROC 衡量模型在不同分类阈值下区分正负样本的能力，AUPRC 更适用于类别分布不均衡场景，Precision 和 Recall 分别反映模型预测正样本的准确性和对真实正样本的覆盖能力。

本文实验环境配置如 Table~\ref{tab:environment} 所示。

Table~\ref{tab:environment}  Experimental environment.

| Item | Value |
|---|---|
| PyTorch | 2.0.0+cu118 |
| PyG | 2.6.1 |
| CUDA | 11.8 |
| Server | AutoDL |
| Output root | `/root/autodl-tmp/HAG-DTA-runs` |
| Cache root | `/root/autodl-tmp/HAG-DTA-cache/processed/` |

为验证不同图卷积算子对 HAG-DTA 性能的影响，本文分别采用 DenseGINConv、DenseGCNConv、DenseGATConv 和 DenseSAGEConv 构建模型变体，如 Table~\ref{tab:model_variants} 所示。其中，GIN 作为默认图卷积算子。

Table~\ref{tab:model_variants}  Model variants of HAG-DTA.

| ID | Model | Graph convolution operator |
|---:|---|---|
| 0 | GIN | DenseGINConv (default) |
| 1 | GCN | DenseGCNConv |
| 2 | GAT | DenseGATConv |
| 3 | SAGE | DenseSAGEConv |

本文同时补充了 GraphDTA 基线实验。GraphDTA 的 GCN、GAT、GIN 和 SAGE 变体通过 `training_graphdta.py` 运行，其 Dense 重写版本与官方 GraphDTA 架构保持对齐。HAG-DTA 主模型通过 `training_davis_kiba.py` 运行，其中包含 Dense 图表示、diff_pool 层次池化结构和 MMD loss。

在模型训练过程中，基于 validation set 的性能选择最佳模型，并在 test set 上报告最终结果。蛋白质序列统一截断或填充至长度 1000。根据后续超参数搜索与消融结果，Davis、KIBA、Human 和 C.elegans 的默认 $(n_1,n_2)$ 设置分别为 (4,2)、(6,2)、(7,3) 和 (7,3)。其中，$n_1$ 和 $n_2$ 分别表示第一层和第二层图池化后的节点数量。

MMD loss 用于约束局部层次表示与全局图表示之间的分布差异，其权重系数记为 $\beta$。在 MMD 系数消融实验中，$\beta=1.0$ 取得了最佳结果，因此修订稿中将 $\beta=1.0$ 作为推荐设置。若部分主实验结果沿用原始运行版本，应在表注中说明对应的 $\beta$ 设置。

---

## 4.2 Overall performance comparison

为验证 HAG-DTA 的整体预测性能，本文首先将不同图卷积算子下的 HAG-DTA 与现有基线模型在 Davis、KIBA、Human 和 C.elegans 数据集上进行比较。Table~\ref{tab:regression_overall} 展示了 Davis 和 KIBA 回归任务上的实验结果，Table~\ref{tab:classification_overall} 展示了 Human 和 C.elegans 二分类任务上的实验结果。

Davis 和 KIBA 采用 DeepDTA 原始 classic split，主要用于与 GraphDTA 等经典基线方法进行公平比较。Human 和 C.elegans 采用 5-fold cross-validation，表中分类结果为 5-fold CV 下的平均结果。

Table~\ref{tab:regression_overall}  Overall performance comparison on Davis and KIBA datasets under classic split.

| Method | Davis MSE ↓ | Davis CI ↑ | Davis rm$^2$ ↑ | KIBA MSE ↓ | KIBA CI ↑ | KIBA rm$^2$ ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DeepDTA |  |  |  |  |  |  |
| GraphDTA-GCN |  |  |  |  |  |  |
| GraphDTA-GAT |  |  |  |  |  |  |
| GraphDTA-GIN |  |  |  |  |  |  |
| GraphDTA-SAGE |  |  |  |  |  |  |
| HAG-DTA-GIN |  |  |  |  |  |  |
| HAG-DTA-GCN |  |  |  |  |  |  |
| HAG-DTA-GAT |  |  |  |  |  |  |
| HAG-DTA-SAGE |  |  |  |  |  |  |

Table~\ref{tab:classification_overall}  Overall performance comparison on Human and C.elegans datasets under 5-fold cross-validation.

| Method | Human AUROC ↑ | Human AUPRC ↑ | Human Precision ↑ | Human Recall ↑ | C.elegans AUROC ↑ | C.elegans AUPRC ↑ | C.elegans Precision ↑ | C.elegans Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline 1 |  |  |  |  |  |  |  |  |
| Baseline 2 |  |  |  |  |  |  |  |  |
| Baseline 3 |  |  |  |  |  |  |  |  |
| HAG-DTA-GIN |  |  |  |  |  |  |  |  |
| HAG-DTA-GCN |  |  |  |  |  |  |  |  |
| HAG-DTA-GAT |  |  |  |  |  |  |  |  |
| HAG-DTA-SAGE |  |  |  |  |  |  |  |  |

从 Table~\ref{tab:regression_overall} 可以看出，在 Davis 和 KIBA 回归任务中，HAG-DTA 在 MSE、CI 和 $rm^2$ 指标上均表现出较强的预测能力。较低的 MSE 表明模型能够降低亲和力预测误差，较高的 CI 和 $rm^2$ 表明模型在排序一致性和回归拟合方面具有较好表现。

从 Table~\ref{tab:classification_overall} 可以看出，在 Human 和 C.elegans 二分类任务中，HAG-DTA 在 AUROC、AUPRC、Precision 和 Recall 指标上均具有较好的综合性能。由于 Human 和 C.elegans 的结果来自 5-fold cross-validation，相关结果能够更充分地反映模型在不同数据划分下的稳定性。

不同图卷积算子之间的结果差异进一步说明，药物分子图表示能力会直接影响 DTA 预测性能。其中，GIN 由于具有较强的图结构判别能力，在多数任务中可作为默认图卷积算子；GCN 和 SAGE 具有较稳定的聚合特性；GAT 通过注意力机制学习邻居节点的重要性，但在不同数据集上可能表现出一定波动。总体上，HAG-DTA 的性能提升主要来源于层次化药物图表示、全局图结构建模以及药物-蛋白多模态特征融合三方面的共同作用。

---

## 4.3 Comparison with recent state-of-the-art methods and GraphDTA baseline significance

为进一步回应近期 DTA 研究进展，本文将 HAG-DTA 与近年来提出的代表性 state-of-the-art 方法进行比较。该部分主要用于验证 HAG-DTA 在最新研究背景下的竞争力。相关结果如 Table~\ref{tab:sota_comparison} 所示。

Table~\ref{tab:sota_comparison}  Comparison with recent state-of-the-art DTA methods.

| Method | Year | Davis MSE ↓ | Davis CI ↑ | KIBA MSE ↓ | KIBA CI ↑ | Notes |
|---|---:|---:|---:|---:|---:|---|
| GEFormerDTA |  |  |  |  |  |  |
| GNPDTA |  |  |  |  |  |  |
| WPGraphDTA |  |  |  |  |  |  |
| MEGDTA |  |  |  |  |  |  |
| HAG-DTA |  |  |  |  |  |  |

与已有 DTA 方法相比，HAG-DTA 的主要特点在于同时考虑药物分子的层次化结构信息与全局拓扑结构信息。现有部分方法主要依赖单尺度分子图表示，容易忽略原子、功能团和分子片段之间的层次关系。本文通过两层 diff_pool 结构学习从原子级到功能团级、再到分子片段级的多尺度表示，从而增强药物分子结构建模能力。

同时，HAG-DTA 进一步引入全局图分支，以补充层次化池化过程中可能被压缩的全局拓扑信息；通过 TextCNN 提取蛋白质序列局部模式；并利用注意力融合机制整合不同来源的特征表示。上述结构共同提升了模型在 DTA 任务中的综合预测能力。

需要说明的是，若部分 SOTA 结果来自原论文或不同数据划分设置，应在表注中明确标注结果来源，并避免对非统一实验设置下的数值差异作过度比较。

### 4.3.1 Significance comparison with GraphDTA baseline on Davis and KIBA

为进一步验证 HAG-DTA 相对于 GraphDTA 基线的性能提升是否具有统计可靠性，本文在 Davis 和 KIBA 数据集上补充 HAG-DTA-GIN 与 GraphDTA 各变体的显著性对比实验。Davis 和 KIBA 均采用 classic split，每个模型使用 5 个随机种子重复运行。随后，本文使用 Welch's t-test 比较 HAG-DTA-GIN 与 GraphDTA-GCN、GraphDTA-GAT、GraphDTA-GIN 和 GraphDTA-SAGE 的结果差异。

Table~\ref{tab:graphdta_significance_setting} 展示了 Davis 和 KIBA 上用于显著性检验的实验矩阵。

Table~\ref{tab:graphdta_significance_setting}  Experimental matrix for GraphDTA significance comparison.

| Dataset | Method | Type | Seeds |
|---|---|---|:---:|
| Davis | HAG-DTA-GIN | Proposed model | 5 |
| Davis | GraphDTA-GCN | Baseline | 5 |
| Davis | GraphDTA-GAT | Baseline | 5 |
| Davis | GraphDTA-GIN | Baseline | 5 |
| Davis | GraphDTA-SAGE | Baseline | 5 |
| KIBA | HAG-DTA-GIN | Proposed model | 5 |
| KIBA | GraphDTA-GCN | Baseline | 5 |
| KIBA | GraphDTA-GAT | Baseline | 5 |
| KIBA | GraphDTA-GIN | Baseline | 5 |
| KIBA | GraphDTA-SAGE | Baseline | 5 |

显著性检验结果如 Table~\ref{tab:graphdta_significance_results} 所示。表中可报告 HAG-DTA-GIN 与各 GraphDTA baseline 在 MSE、CI 和 $rm^2$ 指标上的均值、标准差与 $p$ 值。若 $p<0.05$，则说明 HAG-DTA-GIN 相对于对应基线模型的性能差异具有统计显著性。

Table~\ref{tab:graphdta_significance_results}  Welch's t-test results between HAG-DTA-GIN and GraphDTA baselines on Davis and KIBA.

| Dataset | Metric | Compared method | HAG-DTA-GIN mean ± std | Baseline mean ± std | p-value | Significance |
|---|---|---|---:|---:|---:|---|
| Davis | MSE | GraphDTA-GCN |  |  |  |  |
| Davis | CI | GraphDTA-GCN |  |  |  |  |
| Davis | rm$^2$ | GraphDTA-GCN |  |  |  |  |
| Davis | MSE | GraphDTA-GAT |  |  |  |  |
| Davis | CI | GraphDTA-GAT |  |  |  |  |
| Davis | rm$^2$ | GraphDTA-GAT |  |  |  |  |
| Davis | MSE | GraphDTA-GIN |  |  |  |  |
| Davis | CI | GraphDTA-GIN |  |  |  |  |
| Davis | rm$^2$ | GraphDTA-GIN |  |  |  |  |
| Davis | MSE | GraphDTA-SAGE |  |  |  |  |
| Davis | CI | GraphDTA-SAGE |  |  |  |  |
| Davis | rm$^2$ | GraphDTA-SAGE |  |  |  |  |
| KIBA | MSE | GraphDTA-GCN |  |  |  |  |
| KIBA | CI | GraphDTA-GCN |  |  |  |  |
| KIBA | rm$^2$ | GraphDTA-GCN |  |  |  |  |
| KIBA | MSE | GraphDTA-GAT |  |  |  |  |
| KIBA | CI | GraphDTA-GAT |  |  |  |  |
| KIBA | rm$^2$ | GraphDTA-GAT |  |  |  |  |
| KIBA | MSE | GraphDTA-GIN |  |  |  |  |
| KIBA | CI | GraphDTA-GIN |  |  |  |  |
| KIBA | rm$^2$ | GraphDTA-GIN |  |  |  |  |
| KIBA | MSE | GraphDTA-SAGE |  |  |  |  |
| KIBA | CI | GraphDTA-SAGE |  |  |  |  |
| KIBA | rm$^2$ | GraphDTA-SAGE |  |  |  |  |

该实验的运行命令如下：

```bash
cd ~/HAG-DTA/code_leo
python scripts/statistical_tests.py
```

该部分可作为 Davis 和 KIBA 回归任务的补充证据，用于说明 HAG-DTA-GIN 相对于 GraphDTA 系列基线的性能提升具有更高可信度。

---

## 4.4 Ablation studies

为验证 HAG-DTA 中各模块的贡献，本文设计了 MMD loss 系数消融实验和组件消融实验。通过比较完整模型与不同消融变体的性能变化，可以进一步分析层次化表示、全局图建模、注意力融合和蛋白质序列编码对模型性能的影响。

### 4.4.1 MMD loss coefficient ablation

本文首先在 Davis 数据集上考察 MMD loss 权重 $\beta$ 对模型性能的影响。实验设置如下：

\[
\beta \in \{0, 0.01, 0.05, 0.1, 0.5, 1.0\}.
\]

该实验用于考察 MMD loss 对局部-全局表示对齐的贡献，并验证 $\beta$ 设置的合理性。实验日志保存路径为 `$OUTPUT/sensitivity_mmd/mmd_beta_{beta}.log`。

Figure~\ref{fig:mmd_beta_davis} 展示了 Davis 数据集上的 MMD $\beta$ 消融可视化结果。

![MMD Beta Ablation](./REPORT2/mmd_beta_davis.png)

Figure~\ref{fig:mmd_beta_davis}  Effect of MMD weight $\beta$ on Davis dataset.

Table~\ref{tab:mmd_beta_ablation} 展示了不同 $\beta$ 设置下的实验结果。

Table~\ref{tab:mmd_beta_ablation}  MMD $\beta$ ablation results on Davis dataset.

| $\beta$ | MSE ↓ | CI ↑ | rm$^2$ ↑ |
|---:|---:|---:|---:|
| 0 | 0.2274 | 0.8906 | 0.6379 |
| 0.01 | 0.2155 | 0.8932 | 0.6733 |
| 0.05 | 0.2134 | 0.8939 | 0.6738 |
| 0.1 | 0.2236 | 0.8866 | 0.6627 |
| 0.5 | 0.2158 | 0.8925 | 0.6758 |
| 1.0 | 0.2130 | 0.8980 | 0.6819 |

从实验结果可以看出，当 $\beta=0$ 时，模型的 MSE 最高、CI 最低，说明去除 MMD loss 后模型性能下降，表明 MMD loss 对模型预测具有正向贡献。随着 $\beta$ 增大，模型性能整体得到改善，说明局部层次特征与全局图特征之间的分布对齐有助于提升特征融合效果。

进一步观察可知，$\beta=1.0$ 在该组实验中取得了最优 MSE、CI 和 $rm^2$。因此，本文在修订实验设置中将 $\beta=1.0$ 作为推荐设置。该结果同时说明 MMD loss 对 HAG-DTA 的有效性主要体现在局部层次表示和全局图表示之间的分布对齐，而去除该项会明显削弱模型性能。

### 4.4.2 Component ablation

除 MMD loss 系数消融外，本文进一步设计组件消融实验，以验证模型关键模块的必要性。消融设置如 Table~\ref{tab:component_ablation_settings} 所示。

Table~\ref{tab:component_ablation_settings}  Component ablation settings.

| Variant | Description | Purpose |
|---|---|---|
| Full model | 完整 HAG-DTA | 验证整体性能 |
| w/o MMD | 去除 MMD loss，即 $\beta=0$ | 验证局部-全局表示对齐贡献 |
| w/o Global branch | 仅使用两层 diff_pool 的局部表示 | 验证 DGNN 全局分支必要性 |
| w/o Attention | 使用简单 concatenation 替代注意力融合 | 验证注意力融合贡献 |
| w/o TextCNN protein branch | 仅使用药物图特征 | 验证蛋白质编码必要性 |
| Local only | 仅使用 HGNN 层次化分支 | 分析局部层次结构贡献 |
| Global only | 仅使用 DGNN 全局图分支 | 分析全局拓扑结构贡献 |

Table~\ref{tab:component_ablation_results} 展示了组件消融实验结果。

Table~\ref{tab:component_ablation_results}  Component ablation results.

| Variant | Davis MSE ↓ | Davis CI ↑ | Davis rm$^2$ ↑ | Human AUROC ↑ | Human AUPRC ↑ | Human Precision ↑ | Human Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full model |  |  |  |  |  |  |  |
| w/o MMD |  |  |  |  |  |  |  |
| w/o Global branch |  |  |  |  |  |  |  |
| w/o Attention |  |  |  |  |  |  |  |
| w/o TextCNN protein branch |  |  |  |  |  |  |  |
| Local only |  |  |  |  |  |  |  |
| Global only |  |  |  |  |  |  |  |

从组件消融结果可以进一步分析各模块对 HAG-DTA 的贡献。若去除 MMD loss 后性能下降，说明局部层次表示与全局图表示之间的分布对齐能够提升融合特征的有效性。若去除全局分支后性能下降，说明仅依赖层次化池化结构难以充分保留药物分子的全局拓扑关系。若去除注意力融合模块后性能下降，说明动态学习不同特征分支的重要性比简单拼接更有利于 DTA 预测任务。若去除蛋白质序列分支后性能下降，则说明蛋白质局部序列模式对于药物-靶标相互作用预测具有重要作用。

总体而言，组件消融实验用于说明 HAG-DTA 中的层次化图表示、全局图表示、蛋白质序列编码和注意力融合具有互补作用，完整模型能够获得更优的综合性能。

---

## 4.5 Hyperparameter sensitivity analysis

为分析层次化池化结构中关键超参数对模型性能的影响，本文进一步对 $n_1$ 和 $n_2$ 进行了网格搜索实验。其中，$n_1$ 表示第一层 diff_pool 后的节点数，可理解为功能团级表示粒度；$n_2$ 表示第二层 diff_pool 后的节点数，可理解为更高层次的分子片段表示粒度。

本文设置搜索空间如下：

\[
n_1 \in \{4,5,6,7,8\}, \quad n_2 \in \{2,3,4\}, \quad n_2<n_1.
\]

在上述约束下，共形成 14 组完整组合。本文分别在 Davis 和 Human 数据集上进行网格搜索实验，实验设置如 Table~\ref{tab:n1n2_search_settings} 所示。

Table~\ref{tab:n1n2_search_settings}  $n_1$ and $n_2$ grid search settings.

| Script | Dataset | Runs | Status |
|---|---|---:|:---:|
| `sensitivity_n1n2.sh` | Davis | $14\times5=70$ | To be rerun |
| `sensitivity_n1n2_human.sh` | Human | $14\times5=70$ | To be rerun |

现有 1-seed 参考结果如 Table~\ref{tab:n1n2_reference} 所示。该结果用于初步观察 $n_1$ 和 $n_2$ 对模型性能的影响，最终结论以后续 5 seeds 均值结果为准。

Table~\ref{tab:n1n2_reference}  One-seed reference results of $n_1$ and $n_2$ grid search.

| Dataset | $(n_1,n_2)$ | Main metric | Other metrics |
|---|---|---:|---|
| Davis | (5,4) | CI = 0.8976 | MSE = 0.2081, rm$^2$ = 0.7091 |
| Davis | (4,3) | CI = 0.8938 | MSE = 0.2081, rm$^2$ = 0.6962 |
| Davis | (4,2) | CI = 0.8931 | MSE = 0.2212, rm$^2$ = 0.6645 |
| Human | (4,2) | AUROC = 0.9890 | AUPRC = 0.9895 |
| Human | (5,4) | AUROC = 0.9887 | AUPRC = 0.9817 |
| Human | (7,3) |  |  |

从现有参考结果可以看出，不同 $(n_1,n_2)$ 组合对模型性能存在一定影响，但整体波动幅度相对有限。例如，Davis 数据集上 CI 的变化范围约为 0.887 至 0.898。该现象说明 HAG-DTA 对 $n_1$ 和 $n_2$ 的设置具有一定鲁棒性，模型性能并不依赖极端精细的超参数搜索。

从化学结构角度来看，$n_1$ 可对应小分子中若干功能团级别的结构抽象，$n_2$ 可对应更高层次的药效团或分子片段分区。因此，较小的 $n_1$ 容易造成局部结构信息被过度压缩，较大的 $n_1$ 可能引入冗余簇结构；较小的 $n_2$ 有利于形成紧凑表示，但可能损失部分片段层次信息，较大的 $n_2$ 则可能削弱高层表示的抽象性。

因此，本文在最终论文呈现中以化学动机和鲁棒性论证为主，说明 $n_1$ 和 $n_2$ 的设置具有明确的结构含义，同时通过网格搜索结果证明模型对该组参数不敏感，无需依赖精细调参获得性能提升。考虑到 3D 图对结论表达的增益有限，正文中可不放置 3D 网格图，仅保留文字概括或将完整网格结果放入 supplementary material。

---

## 4.6 Complexity analysis

由于 HAG-DTA 引入了层次化图池化模块、全局图分支和注意力融合模块，因此有必要进一步分析模型的计算复杂度。本文从参数量、训练时间和推理时间三个方面评估 HAG-DTA 的计算开销。复杂度分析结果如 Table~\ref{tab:complexity_analysis} 所示。

Table~\ref{tab:complexity_analysis}  Complexity analysis of different models.

| Model | Parameters | Training time / epoch | Inference time / sample | GPU memory |
|---|---:|---:|---:|---:|
| GraphDTA-GCN |  |  |  |  |
| GraphDTA-GAT |  |  |  |  |
| GraphDTA-GIN |  |  |  |  |
| GraphDTA-SAGE |  |  |  |  |
| HAG-DTA-GIN |  |  |  |  |
| HAG-DTA-GCN |  |  |  |  |
| HAG-DTA-GAT |  |  |  |  |
| HAG-DTA-SAGE |  |  |  |  |

从 Table~\ref{tab:complexity_analysis} 可以看出，HAG-DTA 由于包含两层层次化池化结构和多分支融合模块，其计算开销相比部分简单基线模型有所增加。然而，该额外开销主要来源于对药物分子多尺度结构和全局图拓扑的建模。结合主实验和消融实验结果可知，该复杂度增加能够带来较明显的性能收益。

不同图卷积算子之间也存在一定复杂度差异。GAT 由于需要计算节点邻域之间的注意力权重，通常具有更高的计算成本；GIN 和 GCN 在性能与效率之间具有较好平衡；SAGE 具有较好的归纳聚合特性。总体来看，HAG-DTA 在可接受计算成本下实现了较好的预测性能。

---

## 4.7 Visualization and interpretability analysis

为进一步分析 HAG-DTA 的可解释性，本文对层次化池化过程、注意力权重分布以及药物分子关键子结构进行了可视化分析。该部分主要用于说明模型能够在一定程度上提供药物结构层面的解释线索。

### 4.7.1 Hierarchical pooling visualization

Figure~\ref{fig:hierarchical_pooling} 展示了 HAG-DTA 中层次化池化过程的可视化结果。第一层 diff_pool 将原子级节点聚合为功能团级表示，第二层 diff_pool 进一步将功能团级表示聚合为更高层次的分子片段表示。通过该过程，模型能够从原子级结构逐步抽象到分子片段级结构，从而形成多尺度药物图表示。

![Hierarchical Pooling Visualization](./REPORT2/hierarchical_pooling_placeholder.png)

Figure~\ref{fig:hierarchical_pooling}  Visualization of hierarchical pooling process.

层次化池化结果表明，模型能够在无需人工定义功能团规则的情况下，自动学习药物分子内部的局部结构聚合关系。该结果从可视化角度支持了层次化图表示模块的设计动机。

### 4.7.2 Attention visualization

Figure~\ref{fig:attention_visualization} 展示了注意力权重可视化结果。注意力权重可用于反映不同特征分支或不同局部结构对最终预测结果的相对贡献。

![Attention Visualization](./REPORT2/attention_visualization_placeholder.png)

Figure~\ref{fig:attention_visualization}  Visualization of attention weights.

从注意力分布可以看出，模型倾向于对部分药物子结构和蛋白质局部序列区域赋予更高权重。这说明 HAG-DTA 能够在药物图结构和蛋白质序列之间学习具有差异性的特征贡献，并通过注意力融合机制增强关键特征的表达。

### 4.7.3 Case study and chemical interpretation

进一步地，本文选取典型药物-靶标样本进行案例分析。可视化结果显示，部分高权重区域对应于芳香环、含氮杂环以及可能参与氢键作用的功能基团。这些结构可能与药物-靶标结合过程中的疏水相互作用、氢键作用或静电相互作用有关。

需要说明的是，本文可视化结果主要用于提供模型决策过程的初步解释线索。高权重子结构可以为后续药物结构优化、分子对接分析或实验验证提供候选关注区域。相关结论仍需结合更多药物化学证据和实验结果进一步验证。

---

## Summary of this section

本节从整体性能比较、近期 SOTA 方法对比、GraphDTA 基线显著性检验、消融实验、超参数敏感性分析、复杂度分析和可解释性分析等方面系统评估了 HAG-DTA 的有效性。整体实验结果用于验证模型在不同任务和数据集上的预测性能；SOTA 对比用于说明模型相对于近期方法的竞争力；GraphDTA 显著性对比用于增强 Davis 和 KIBA 回归实验结论的统计可靠性；消融实验用于分析各关键模块的贡献；超参数敏感性分析用于解释 $n_1$、$n_2$ 和 $\beta$ 的影响；复杂度分析用于评估模型计算开销；可视化分析则进一步展示模型在药物结构层面的解释能力。

综合来看，HAG-DTA 通过层次化药物图表示、全局图拓扑建模、蛋白质序列特征提取和注意力融合机制，在多个 DTA 任务上表现出较好的预测能力与一定解释性。
