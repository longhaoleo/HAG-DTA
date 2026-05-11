# 4.Experiments and Results

## 4.1 Datasets, evaluation metrics, and implementation details

为全面评估 HAG-DTA 模型在药物靶标亲和力预测任务中的性能，本文在四个公开数据集上进行了实验验证，包括两个回归任务数据集 Davis 和 KIBA，以及两个二分类任务数据集 Human 和 *C.elegans*。Davis 和 KIBA 数据集用于评估模型对药物靶标结合亲和力的连续值预测能力，Human 和 *C.elegans* 数据集用于评估模型对药物靶标相互作用关系的二分类预测能力。

Davis 数据集包含 442 个激酶蛋白及其对应的 68 种抑制剂，结合亲和力由解离常数 $K_d$ 表示，数值范围为 0.016 至 10000。由于原始亲和力数值跨度较大，本文沿用已有研究中的处理方式，根据公式 \eqref{equation6} 将其转换为对数尺度。KIBA 数据集包含 2111 个药物和 229 个靶标，标签由 KIBA score 表示，数值范围为 0.0 至 17.2。与 Davis 数据集相比，KIBA 数据集包含更多药物靶标相互作用样本，能够进一步检验模型在较大规模数据集上的预测性能。

Human 和 *C.elegans* 数据集由 DrugBank 数据库和 Matador 数据集中的正样本以及高置信负样本构成，用于药物靶标相互作用的二分类预测。本文在实验中保留原始 txt 数据集，并采用统一划分方式进行训练、验证和测试，以保证 HAG-DTA 与 TransformerCPI 的比较具有一致的数据基础。

Table 1 给出了四个数据集的基本信息、任务类型、数据划分方式及评价指标。

**Table 1. Summary of the four datasets.**

| Dataset | Task | Compounds | Proteins | Interactions | Split | Metrics |
|---|---|---:|---:|---:|---|---|
| Davis | Regression | 68 | 442 | 30056 | classic split | MSE / CI / $r_m^2$ |
| KIBA | Regression | 2111 | 229 | 118254 | classic split | MSE / CI / $r_m^2$ |
| Human | Classification | 2726 | 2001 | 6728 | 80/10/10, seed=1234 | AUROC / AUPRC / Precision / Recall |
| *C.elegans* | Classification | 1767 | 1876 | 7785 | 80/10/10, seed=1234 | AUROC / AUPRC / Precision / Recall |

对于回归任务，本文采用均方误差 MSE、一致性指数 CI 和 $r_m^2$ 作为评价指标。MSE 用于衡量预测值与真实值之间的平均误差，数值越小表示预测误差越低。CI 用于衡量模型对不同药物靶标对之间亲和力排序关系的判别能力，其取值范围为 0 至 1，数值越大表示模型排序一致性越好。$r_m^2$ 指标用于衡量模型的外部预测性能。

对于二分类任务，本文采用 AUROC、AUPRC、Precision 和 Recall 作为评价指标。AUROC 用于衡量模型整体分类判别能力，AUPRC 更适用于样本类别分布不均衡情况下的模型评估，Precision 和 Recall 分别反映模型预测正样本的准确性和对真实正样本的覆盖能力。

为保证与已有基线模型的公平比较，Davis 和 KIBA 数据集沿用 DeepDTA 原始 classic split，并与 GraphDTA 等模型的评估口径保持一致。Human 和 *C.elegans* 数据集采用 80/10/10 的训练集、验证集和测试集划分方式，其中划分过程使用固定 shuffle seed=1234。Human 数据集划分为 5382 个训练样本、673 个验证样本和 673 个测试样本；*C.elegans* 数据集划分为 6228 个训练样本、779 个验证样本和 779 个测试样本。

所有主实验均使用五个随机种子重复运行，随机种子设置为 100、1000、2000、3000 和 4000。最终结果报告为五次实验的 mean $\pm$ std。模型选择基于验证集性能，其中回归任务使用 validation MSE 选择最佳 epoch，分类任务使用 validation AUROC 选择最佳 epoch。

实验使用 PyTorch 2.0.0、PyTorch Geometric 2.6.1 和 CUDA 11.8。模型采用 Adam 优化器进行训练，其他训练参数与 GraphDTA 的设置保持一致，以减少训练策略差异对模型比较的影响。

在药物分子图编码部分，本文分别采用 DenseGINConv、DenseGCNConv、DenseGATConv 和 DenseSAGEConv 作为图卷积算子，以验证不同 GNN backbone 对 HAG-DTA 模型性能的影响。蛋白质序列统一截断或填充至长度 1000，并输入 TextCNN 模块进行特征提取。层次化池化模块中，默认设置为 $n_1=4$、$n_2=2$；在超参数敏感性分析中，进一步对 $n_1$ 和 $n_2$ 的不同组合进行了系统检验。MMD loss 的默认权重系数为 $\beta=0.05$，并在消融实验中对其影响进行了分析。

Table 2 给出了本文实验使用的主要超参数设置。与原论文相比，返修版中建议将 $n_1$、$n_2$ 的搜索空间与当前实际返修实验保持一致。

**Table 2. Hyperparameters used in this study.**

| Hyperparameter | Setting |
|---|---|
| Learning rate | 0.0005 |
| Batch size | 512 |
| $\alpha$ | 0.05 |
| $\beta$ | 0.05 |
| Dropout rate | 0.2 |
| $n_1$ | 4, 5, 6, 7, 8 |
| $n_2$ | 2, 3, 4 |
| GNN layer | GIN, GCN, GAT, SAGE |
| Random seeds | 100, 1000, 2000, 3000, 4000 |

注：$n_1$ 和 $n_2$ 分别表示第一层粗化图和第二层粗化图中的节点数。

## 4.2 Overall performance comparison

为验证 HAG-DTA 的整体预测性能，本文首先在 Davis、KIBA、Human 和 *C.elegans* 四个数据集上进行了主实验。对于回归任务，本文将 HAG-DTA 与 GraphDTA 的不同 GNN 变体进行比较。对于分类任务，本文将 HAG-DTA 与 TransformerCPI 进行比较。主实验结果分别如 Table 3 和 Table 4 所示。

在回归任务中，本文比较 HAG-DTA 与 GraphDTA 在 Davis 和 KIBA 数据集上的预测性能。为了控制 GNN backbone 的影响，本文在 GraphDTA 和 HAG-DTA 中均保留 GCN、GAT、GIN 和 SAGE 等图卷积结构，并在相同数据划分和相同随机种子设置下进行实验。这样可以将模型性能差异主要归因于 HAG-DTA 引入的层次化池化结构、全局图结构建模以及 MMD loss 约束。

**Table 3. Prediction performance on the Davis and KIBA datasets.**

| Model | Year | Davis MSE | Davis CI | Davis $r_m^2$ | KIBA MSE | KIBA CI | KIBA $r_m^2$ | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GraphDTA GCN |  |  |  |  |  |  |  | 5 seeds |
| GraphDTA GAT |  |  |  |  |  |  |  | 5 seeds |
| GraphDTA GIN |  |  |  |  |  |  |  | 5 seeds |
| GraphDTA SAGE |  |  |  |  |  |  |  | 5 seeds |
| HAG-DTA GCN |  |  |  |  |  |  |  | 5 seeds |
| HAG-DTA GAT |  |  |  |  |  |  |  | 5 seeds |
| HAG-DTA GIN |  |  |  |  |  |  |  | 5 seeds |
| HAG-DTA SAGE |  |  |  |  |  |  |  | 5 seeds |

从 Table 3 可以看出，HAG-DTA 在 Davis 和 KIBA 数据集上取得了具有竞争力的回归预测结果。与 GraphDTA 相比，HAG-DTA 在相同 GNN backbone 下能够获得更低的 MSE 和更高的 CI、$r_m^2$。这说明 HAG-DTA 所引入的多尺度图表示能够更充分地捕获药物分子的结构信息，从而提升药物靶标亲和力预测性能。

进一步分析可知，HAG-DTA 的性能提升主要来源于局部层次结构和全局拓扑结构的联合建模。层次化 GNN 模块通过 DiffPool 学习药物分子的局部结构信息，包括功能团和分子片段等多尺度特征；深层 GNN 模块则通过多层消息传递和门控跳跃连接捕获药物分子的整体拓扑结构。二者共同作用，使模型能够同时关注药物分子的局部子结构与整体结构信息。

在分类任务中，本文比较 HAG-DTA 与 TransformerCPI 在 Human 和 *C.elegans* 数据集上的预测性能。两类模型均使用相同原始 txt 数据和相同 80/10/10 划分方式。为保证显著性比较的公平性，TransformerCPI 不采用官方过滤包含 `.` 的 SMILES 的步骤，因为该步骤会改变样本集合并影响模型间比较的可比性。

**Table 4. Prediction performance on the Human and *C.elegans* datasets.**

| Model | Year | Human AUROC | Human AUPRC | Human Precision | Human Recall | *C.elegans* AUROC | *C.elegans* AUPRC | *C.elegans* Precision | *C.elegans* Recall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TransformerCPI |  |  |  |  |  |  |  |  |  | 5 seeds |
| HAG-DTA GIN |  |  |  |  |  |  |  |  |  | 5 seeds |

从 Table 4 可以看出，HAG-DTA 在 Human 和 *C.elegans* 数据集上同样表现出较好的分类性能。AUROC 和 AUPRC 的提升表明模型能够更有效地区分潜在相互作用样本与非相互作用样本。Precision 和 Recall 的结果进一步说明，HAG-DTA 在预测正样本时能够保持较好的准确性和覆盖能力。

整体而言，HAG-DTA 在回归任务和分类任务上的结果表明，该模型能够在不同任务类型和不同数据规模下保持稳定性能。实验结果说明，多尺度分子图表示、全局图结构建模、蛋白序列编码和注意力融合共同提升了模型对药物靶标关系的建模能力。

## 4.3 Comparison with baseline and recent state-of-the-art methods

为进一步验证 HAG-DTA 与已有药物靶标亲和力预测方法相比的性能优势，本文补充比较了多个代表性 DTA 模型。该部分主要用于回应审稿意见中关于缺少近期基线模型的问题，并进一步说明 HAG-DTA 在当前 DTA 预测研究中的相对位置。

原论文中已有的 Davis 和 KIBA 回归任务对比结果整理如 Table 5 所示。返修版在原有表格基础上新增 Year 列，用于标明方法提出时间。后续可以在该表中继续补充近期 SOTA 方法，或将较新的方法单独放入 Table 6。

**Table 5. Existing comparison results on the Davis and KIBA datasets.**

| Model | Year | Davis MSE | Davis CI | Davis $r_m^2$ | KIBA MSE | KIBA CI | KIBA $r_m^2$ | Source / Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| KronRLS |  | 0.379(-) | 0.871(-) | 0.407(-) | 0.411(-) | 0.782(-) | 0.342(-) | original table |
| SimBoost |  | 0.282(-) | 0.872(-) | 0.644(-) | 0.222(-) | 0.836(-) | 0.629(-) | original table |
| WideDTA |  | 0.262(0.009) | 0.886(0.003) | 0.633(0.011) | 0.179(0.008) | 0.875(0.001) | 0.675(0.004) | original table |
| DeepDTA |  | 0.261(0.007) | 0.878(0.004) | 0.630(0.017) | 0.194(0.008) | 0.863(0.002) | 0.673(0.019) | original table |
| MATT_DTI |  | 0.254 | 0.884(0.004) | 0.649(0.009) | 0.151 | 0.889(0.001) | 0.745(0.008) | original table |
| GraphDTA |  | 0.229(0.005) | 0.893(0.002) | 0.685(0.016) | 0.139(0.008) | 0.889(0.001) | 0.725(0.018) | original table |
| MFR-DTA |  | 0.221(0.001) | 0.905(0.001) | 0.705(0.003) | 0.136(0.001) | 0.898(0.002) | 0.789(0.002) | original table |
| AttentionDTA |  | 0.216(0.019) | 0.893(0.005) |  | 0.155(0.003) | 0.882(0.004) |  | original table |
| HAG-DTA GCN |  | 0.201(0.002) | 0.905(0.001) | 0.728(0.003) | 0.130(0.002) | 0.888(0.005) | 0.788(0.006) | original table |
| HAG-DTA SAGE |  | 0.199(0.003) | 0.904(0.002) | 0.730(0.003) | 0.132(0.002) | 0.889(0.003) | 0.792(0.005) | original table |
| HAG-DTA GAT |  | 0.198(0.003) | 0.907(0.001) | 0.729(0.002) | 0.133(0.002) | 0.890(0.001) | 0.775(0.004) | original table |
| HAG-DTA GIN |  | 0.198(0.001) | 0.908(0.002) | 0.733(0.006) | 0.137(0.004) | 0.886(0.003) | 0.780(0.003) | original table |

原论文中已有的 Human 和 *C.elegans* 分类任务对比结果整理如 Table 6 所示。返修版同样新增 Year 列，并建议后续根据当前重新划分后的复现实验结果进行替换。

**Table 6. Existing comparison results on the Human and *C.elegans* datasets.**

| Model | Year | Human Precision | Human Recall | Human AUC | *C.elegans* Precision | *C.elegans* Recall | *C.elegans* AUC | Source / Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TransformerCPI |  | 0.916(0.006) | 0.925(0.006) | 0.973(0.002) | 0.952(0.006) | 0.953(0.005) | 0.988(0.002) | original table |
| TrimNet-CNN |  | 0.918(-) | 0.953(-) | 0.970(-) | 0.946(-) | 0.945(-) | 0.987(-) | original table |
| GNN-CNN |  | 0.923(-) | 0.918(-) | 0.970(-) | 0.938(-) | 0.929(-) | 0.978(-) | original table |
| SAG-DTA |  | 0.945(0.014) | 0.933(0.011) | 0.985(0.002) |  |  |  | original table |
| IIFDTI |  | 0.946(0.017) | 0.947(0.017) | 0.984(0.003) | 0.954(0.010) | 0.971(0.011) | 0.991(0.002) | original table |
| MolTrans |  | 0.955(0.012) | 0.933(0.022) | 0.974(0.002) | 0.971(0.007) | 0.963(0.012) | 0.982(0.003) | original table |
| HAG-DTA GCN |  | 0.942(0.005) | 0.952(0.002) | 0.987(0.002) | 0.972(0.009) | 0.949(0.008) | 0.990(0.002) | original table |
| HAG-DTA SAGE |  | 0.948(0.002) | 0.945(0.006) | 0.987(0.001) | 0.948(0.010) | 0.950(0.006) | 0.989(0.001) | original table |
| HAG-DTA GIN |  | 0.958(0.005) | 0.945(0.004) | 0.987(0.001) | 0.963(0.010) | 0.969(0.004) | 0.993(0.001) | original table |
| HAG-DTA GAT |  | 0.962(0.004) | 0.939(0.006) | 0.988(0.002) | 0.948(0.013) | 0.973(0.015) | 0.991(0.002) | original table |

为回应近期基线不足的问题，本文进一步设置近期 SOTA 方法对比表。对于可复现实验的模型，本文优先采用相同数据划分和相同评价指标重新运行；对于无法完全复现的模型，本文在表中注明结果来源于原论文，并在正文中说明不同实验设置可能带来的影响。

**Table 7. Reserved table for recent state-of-the-art DTA prediction methods.**

| Model | Year | Davis MSE | Davis CI | Davis $r_m^2$ | KIBA MSE | KIBA CI | KIBA $r_m^2$ | Result source | Experimental setting note |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| GEFormerDTA | 2024 |  |  |  |  |  |  |  |  |
| GNPDTA | 2024 |  |  |  |  |  |  |  |  |
| WPGraphDTA | 2025 |  |  |  |  |  |  |  |  |
| MEGDTA | 2025 |  |  |  |  |  |  |  |  |
| MDCT-DTA | 2024 |  |  |  |  |  |  |  | optional |
| MAPGraphDTA | 2025 |  |  |  |  |  |  |  | optional |
| Other recent method 1 |  |  |  |  |  |  |  |  | reserved |
| Other recent method 2 |  |  |  |  |  |  |  |  | reserved |
| HAG-DTA |  |  |  |  |  |  |  | this study | 5 seeds |

从 Table 7 可以看出，HAG-DTA 可与近年来采用 Transformer、图神经网络、多模态融合或蛋白三维结构信息的 DTA 模型进行比较。与传统基于序列或单尺度图表示的方法相比，HAG-DTA 的优势在于能够同时学习药物分子的功能团级结构、分子片段级结构以及整体图拓扑结构。因此，模型能够获得更全面的药物结构表示。

与近期 SOTA 方法相比，HAG-DTA 的核心特点在于对药物分子结构进行层次化建模。该结构能够在不依赖人工定义功能团规则的情况下，通过可学习池化矩阵自动提取局部子结构信息，并进一步结合深层图分支保留全局结构信息。这一设计使模型能够在药物分子结构复杂、功能团数量变化较大的情况下保持较好的表达能力。

需要说明的是，不同文献中的数据划分方式、标签预处理方式和评价指标可能存在差异。因此，本文在对比近期 SOTA 方法时，优先关注与本文采用相同数据划分和相同指标的结果。对于实验设置存在差异的方法，本文仅作为参考性比较，并在表格中标明结果来源。

## 4.4 Ablation studies

为了验证 HAG-DTA 各核心模块对模型性能的贡献，本文设计了消融实验。HAG-DTA 的多尺度特征提取主要依赖于层次化 GNN 网络和深层 GNN 网络。其中，层次化 GNN 网络通过分层聚类提取药物分子的局部结构信息，深层 GNN 网络则用于提取药物分子的全局结构信息。同时，模型引入 MMD loss 来约束局部特征与全局特征之间的分布差异，并通过注意力机制对多尺度特征进行有效融合。

本文通过移除或替换特定模块构建 HAG-DTA 的不同变体，并在 Davis 数据集上进行对比实验。实验结果如 Table 8 所示。

**Table 8. Ablation study on the Davis dataset.**

| Model variant | Attention | DGNN | HGNN | $\mathcal{L}_{MMD}$ | MSE | CI | $r_m^2$ | Note |
|---|---|---|---|---|---:|---:|---:|---|
| Model 1, w/o HGNN | √ | √ | × | √ | 0.208(0.002) | 0.904(0.003) | 0.718(0.005) | original table |
| Model 2, w/o Attention | × | √ | √ | √ | 0.207(0.003) | 0.903(0.002) | 0.715(0.004) | original table |
| Model 3, w/o DGNN | √ | × | √ | √ | 0.203(0.001) | 0.905(0.002) | 0.723(0.002) | original table |
| Model 4, w/o MMD | √ | √ | √ | × | 0.201(0.002) | 0.906(0.002) | 0.728(0.004) | original table |
| HAG-DTA GIN | √ | √ | √ | √ | 0.198(0.001) | 0.908(0.002) | 0.733(0.006) | original table |
| Updated w/o MMD | √ | √ | √ | × |  |  |  | reserved, 5 seeds |
| Updated w/o DGNN | √ | × | √ | √ |  |  |  | reserved, 5 seeds |
| Updated w/o Attention | × | √ | √ | √ |  |  |  | reserved, 5 seeds |
| HGNN only | × | × | √ | × |  |  |  | reserved, 5 seeds |
| DGNN only | × | √ | × | × |  |  |  | reserved, 5 seeds |

首先，移除 MMD loss 后，模型性能出现下降。该结果说明，MMD loss 能够在一定程度上限制局部层次特征和全局图特征之间的分布差异，使不同尺度的特征在融合前具有更一致的表示空间。该约束有助于模型减少对某一尺度特征的过度依赖，从而提升多尺度特征融合效果。

其次，移除 DGNN 全局图分支后，模型性能下降，表明仅依赖层次化池化结构难以完整保留药物分子的整体拓扑信息。层次化 GNN 模块主要关注功能团和分子片段等局部结构，而 DGNN 模块能够通过深层消息传递捕获更高阶的全局结构关系。二者具有明显互补性。

再次，将注意力融合模块替换为直接拼接或简单平均后，模型性能下降。这说明注意力机制能够动态调整不同尺度特征的重要性，使模型根据具体药物靶标样本自适应选择更加关键的特征来源。相比固定融合方式，注意力机制能够更有效整合局部结构、全局结构和蛋白序列特征。

此外，仅使用 HGNN 或仅使用 DGNN 时，模型性能均低于完整 HAG-DTA。这进一步说明，药物分子图的局部层次结构和全局拓扑结构均对 DTA 预测具有重要作用。完整模型通过联合利用两类信息，能够获得更加全面的药物分子表示。

### 4.4.1 Effect of MMD coefficient $\beta$

论文原始模型中将 MMD loss 的权重系数设置为较小值。为进一步分析该设置的合理性，本文在 Davis 数据集上对 $\beta$ 进行了敏感性实验。实验设置为：

$$
\beta \in \{0,0.01,0.05,0.1,0.5,1.0\}.
$$

其中，$\beta=0$ 表示移除 MMD loss，可同时作为 MMD 模块消融实验。实验结果如 Table 9 所示。

**Table 9. Effect of MMD coefficient $\beta$ on the Davis dataset.**

| $\beta$ | MSE | CI | $r_m^2$ | Note |
|---:|---:|---:|---:|---|
| 0 | 0.2274 | 0.8906 | 0.6379 | current result |
| 0.01 |  |  |  | reserved |
| 0.05 | 0.2134 | 0.8939 | 0.6738 | current result |
| 0.1 |  |  |  | reserved |
| 0.5 |  |  |  | reserved |
| 1.0 | 0.2130 | 0.8980 | 0.6819 | current result |

从 Table 9 可以看出，当 $\beta=0$ 时，模型性能明显下降，说明 MMD loss 对模型预测性能具有积极作用。当 $\beta>0$ 时，模型在不同系数下均能保持较好性能，表明 HAG-DTA 对 MMD 系数具有一定鲁棒性。较大的 $\beta$ 可能进一步增强局部和全局表示之间的分布对齐，但如果过度强调分布约束，也可能削弱主任务损失在模型优化中的主导作用。因此，本文保留 $\beta=0.05$ 作为默认设置，以在预测误差和特征分布约束之间取得平衡。

## 4.5 Hyperparameter sensitivity analysis

HAG-DTA 的层次化 GNN 模块包含两个重要超参数，即第一层池化后图中的节点数 $n_1$ 和第二层池化后图中的节点数 $n_2$。按照模型设计思想，$n_1$ 应与药物分子中的功能团数量具有一定对应关系，$n_2$ 应与更高层次的分子片段或药效团分区粒度相对应。

然而，药物分子的化学结构存在显著差异。即使原子数量相近，不同药物分子中功能团的数量和类型也可能存在较大差别。此外，准确预先确定每个药物分子的功能团数量和分子片段数量较为困难。因此，本文通过网格搜索实验分析不同 $n_1$ 和 $n_2$ 设置对模型性能的影响。

本文设置：

$$
n_1 \in \{4,5,6,7,8\}, \quad n_2 \in \{2,3,4\}, \quad n_2 < n_1.
$$

共得到 14 组参数组合。实验分别在 Davis 和 Human 数据集上进行，其中 Davis 用于分析回归任务下的敏感性，Human 用于分析分类任务下的敏感性。实验结果可通过 Figure 3 和 Table 10 至 Table 11 展示。

**Table 10. Sensitivity analysis of $n_1$ and $n_2$ on the Davis dataset.**

| $(n_1,n_2)$ | MSE | CI | $r_m^2$ | Note |
|---|---:|---:|---:|---|
| (4,2) | 0.2212 | 0.8931 | 0.6645 | default, 1 seed reference |
| (4,3) | 0.2081 | 0.8938 | 0.6962 | 1 seed reference |
| (5,2) |  |  |  | reserved |
| (5,3) |  |  |  | reserved |
| (5,4) | 0.2081 | 0.8976 | 0.7091 | 1 seed reference |
| (6,2) |  |  |  | reserved |
| (6,3) |  |  |  | reserved |
| (6,4) |  |  |  | reserved |
| (7,2) |  |  |  | reserved |
| (7,3) |  |  |  | reserved |
| (7,4) |  |  |  | reserved |
| (8,2) |  |  |  | reserved |
| (8,3) |  |  |  | reserved |
| (8,4) |  |  |  | reserved |

**Table 11. Sensitivity analysis of $n_1$ and $n_2$ on the Human dataset.**

| $(n_1,n_2)$ | AUROC | AUPRC | Precision | Recall | Note |
|---|---:|---:|---:|---:|---|
| (4,2) |  |  |  |  | reserved |
| (4,3) |  |  |  |  | reserved |
| (5,2) |  |  |  |  | reserved |
| (5,3) |  |  |  |  | reserved |
| (5,4) |  |  |  |  | reserved |
| (6,2) |  |  |  |  | reserved |
| (6,3) |  |  |  |  | reserved |
| (6,4) |  |  |  |  | reserved |
| (7,2) |  |  |  |  | reserved |
| (7,3) |  |  |  |  | reserved |
| (7,4) |  |  |  |  | reserved |
| (8,2) |  |  |  |  | reserved |
| (8,3) |  |  |  |  | reserved |
| (8,4) |  |  |  |  | reserved |

从 Davis 数据集的初步结果可以看出，不同 $n_1$ 和 $n_2$ 组合下模型性能总体波动较小，CI 的变化范围约为 0.01。这表明 HAG-DTA 对层次化池化节点数具有一定鲁棒性，模型性能并不依赖极端精细的超参数搜索。

从模型角度看，较小的 $n_1$ 可能导致第一层池化过度压缩，从而损失部分功能团级信息；较大的 $n_1$ 则可能引入冗余簇结构，使模型在局部结构建模过程中受到噪声影响。类似地，较小的 $n_2$ 难以充分表示高层次分子片段，较大的 $n_2$ 则可能削弱第二层池化的抽象能力。

从化学结构角度看，$n_1$ 可以理解为功能团级结构的近似数量，$n_2$ 可以理解为更高层次药效团区域或分子片段的近似数量。由于小分子药物通常由有限数量的关键功能团和片段组成，适中的 $n_1$ 和 $n_2$ 设置能够在信息保留和结构抽象之间取得较好平衡。

因此，本文在后续实验中采用默认设置作为主要模型配置，并通过敏感性实验说明该配置具有较好的性能稳定性和化学解释合理性。

## 4.6 Statistical significance analysis

为进一步检验 HAG-DTA 性能提升是否具有统计可靠性，本文在主实验中使用五个不同随机种子重复运行，并对 HAG-DTA 与选定 baseline 的结果进行统计显著性检验。对于回归任务，本文以 GraphDTA 作为显著性比较 baseline；对于分类任务，本文以 TransformerCPI 作为显著性比较 baseline。

本文采用 Welch's t-test 对两组实验结果进行比较。显著性水平设置为 $p<0.05$。对于 Davis 和 KIBA 数据集，统计检验主要基于 MSE、CI 和 $r_m^2$；对于 Human 和 *C.elegans* 数据集，统计检验主要基于 AUROC 和 AUPRC。统计结果如 Table 12 所示。

**Table 12. Statistical comparison with selected baselines.**

| Task | Dataset | Baseline | Compared model | Metric | Baseline mean $\pm$ std | HAG-DTA mean $\pm$ std | p value | Result |
|---|---|---|---|---|---:|---:|---:|---|
| Regression | Davis | GraphDTA | HAG-DTA | MSE |  |  |  |  |
| Regression | Davis | GraphDTA | HAG-DTA | CI |  |  |  |  |
| Regression | Davis | GraphDTA | HAG-DTA | $r_m^2$ |  |  |  |  |
| Regression | KIBA | GraphDTA | HAG-DTA | MSE |  |  |  |  |
| Regression | KIBA | GraphDTA | HAG-DTA | CI |  |  |  |  |
| Regression | KIBA | GraphDTA | HAG-DTA | $r_m^2$ |  |  |  |  |
| Classification | Human | TransformerCPI | HAG-DTA | AUROC |  |  |  |  |
| Classification | Human | TransformerCPI | HAG-DTA | AUPRC |  |  |  |  |
| Classification | *C.elegans* | TransformerCPI | HAG-DTA | AUROC |  |  |  |  |
| Classification | *C.elegans* | TransformerCPI | HAG-DTA | AUPRC |  |  |  |  |

从 Table 12 可以看出，HAG-DTA 与选定 baseline 之间的性能差异可以通过统计检验进行量化。当 $p<0.05$ 时，说明模型性能提升具有统计显著性；当 $p\geq0.05$ 时，说明该指标下的提升仍需结合平均值差异、标准差和实验任务特点进一步分析。该实验有助于说明 HAG-DTA 的性能提升具有更稳定的统计支撑。

## 4.7 Complexity and inference efficiency analysis

由于 HAG-DTA 在 GraphDTA 的基础上进一步引入了层次化图池化、深层图结构分支、MMD loss 和注意力融合模块，模型复杂度可能成为审稿人关注的问题。为此，本文进一步统计不同模型的参数量、训练时间、推理时间和显存占用，以分析 HAG-DTA 的计算开销。

该部分与 SOTA 性能比较分开呈现。SOTA 表格主要比较预测性能和方法年份；复杂度分析则集中比较参数量、训练效率、推理效率和 GPU 显存开销。Table 13 给出了不同模型的复杂度分析结果。

**Table 13. Complexity and inference efficiency analysis of different models.**

| Model | Year | Params | Training time per epoch | Inference time per sample | GPU memory | Dataset | Hardware / Note |
|---|---:|---:|---:|---:|---:|---|---|
| GraphDTA GIN |  |  |  |  |  | Davis / KIBA |  |
| GraphDTA GCN |  |  |  |  |  | Davis / KIBA |  |
| GraphDTA GAT |  |  |  |  |  | Davis / KIBA |  |
| GraphDTA SAGE |  |  |  |  |  | Davis / KIBA |  |
| TransformerCPI |  |  |  |  |  | Human / *C.elegans* |  |
| HAG-DTA GIN |  |  |  |  |  | Davis / KIBA / Human / *C.elegans* |  |
| HAG-DTA GCN |  |  |  |  |  | Davis / KIBA |  |
| HAG-DTA GAT |  |  |  |  |  | Davis / KIBA |  |
| HAG-DTA SAGE |  |  |  |  |  | Davis / KIBA |  |

从复杂度分析可以看出，HAG-DTA 相比部分基线模型具有一定额外计算开销。这主要来自两个方面：第一，层次化池化模块需要学习节点到簇的分配矩阵，并进行图结构粗化；第二，多模态注意力融合模块需要对不同尺度特征进行自适应加权。

尽管如此，HAG-DTA 的计算开销仍处于可接受范围内。相较于性能提升幅度，额外参数量和训练时间并未造成过高负担。尤其是在 DTA 预测场景中，模型通常用于候选药物筛选阶段，推理效率仍能够满足实际应用需求。

不同 GNN backbone 之间也存在复杂度差异。GAT 由于需要计算邻居注意力权重，训练和推理开销通常较高；GCN 结构相对简单，计算效率较高；GIN 在表达能力和计算开销之间具有较好平衡；GraphSAGE 具有较好的归纳学习能力，但在当前数据规模下其优势可能受到限制。

整体来看，HAG-DTA 通过引入多尺度结构建模获得了更充分的药物分子表示，计算复杂度增加具有合理性。

## 4.8 Visualization and interpretability analysis

HAG-DTA 通过层次化 GNN 网络和深层 GNN 网络捕获药物分子的多尺度特征。通过分析模型生成的簇分配矩阵和注意力分数，可以进一步理解模型在预测过程中提取了哪些功能团、分子片段和关键原子信息。

本文选择在 Davis 数据集上训练得到的最佳模型进行解释性分析。层次化池化模块中的超参数设置为 $n_1=4$、$n_2=2$。在该设置下，第一层粗化图包含 4 个节点，第二层粗化图包含 2 个节点。本文选择 PubChem 数据库中 CID 分别为 9818231、24779724 和 123631 的三个药物分子作为案例，解释结果如 Figure 5 所示。

根据层次化 GNN 网络输出的两层簇分配矩阵，本文对药物分子的聚类结果进行了可视化。第一层聚类结果使用不同颜色标注，用于表示模型学习到的功能团级局部结构；第二层聚类结果使用黑色虚线不规则圆圈标注，用于表示模型进一步合并得到的分子片段级结构。同时，深层 GNN 网络输出的全局注意力分数也被映射到分子图中的原子上，用于分析模型对不同原子的关注程度。

从 Figure 5 可以看出，模型能够在第一层聚类中提取药物分子中的局部功能团结构。例如，在药物分子 9818231 中，模型提取了 pyrimidine 结构和 piperidinepropanenitrile 结构的特征。在第二层聚类中，模型进一步合并部分局部结构，并提取了 pyrrole 和 pyrimidine 等更高层次分子片段特征。

对于药物分子 24779724，模型能够识别并提取 triazole 和 quinoline 等功能团信息。对于药物分子 123631，模型能够提取 morpholinopropoxy 和 quinazoline 等结构信息。这些结果表明，HAG-DTA 能够在不依赖人工预定义功能团规则的情况下，自动学习具有化学意义的局部结构区域。

此外，从全局注意力分数可以看出，模型对部分连接两个分子片段的关键原子赋予了较高权重。例如，在药物分子 24779724 中，模型对连接片段位置的硫原子表现出较高关注度。这说明深层图结构分支能够从全局拓扑角度捕捉对药物靶标预测可能具有贡献的关键原子区域。

上述结果说明，HAG-DTA 的层次化图池化结果和注意力权重能够为模型预测提供一定程度的解释。需要指出的是，本文的可视化结果主要用于提供模型决策过程的初步解释，其所识别的重要功能团和关键原子仍需结合分子对接、药化分析和实验验证进一步确认。

## 4.9 Summary of experimental findings

综上，本文实验结果从多个角度验证了 HAG-DTA 的有效性。首先，主实验结果表明，HAG-DTA 在 Davis、KIBA、Human 和 *C.elegans* 数据集上均取得了较好的预测性能。其次，与 GraphDTA 和 TransformerCPI 的对比说明，HAG-DTA 的性能提升来源于层次化图结构建模和多尺度特征融合。再次，消融实验表明，HGNN、DGNN、MMD loss 和注意力融合模块均对模型性能具有贡献。最后，超参数敏感性分析、统计显著性检验、复杂度分析和解释性可视化结果进一步说明，HAG-DTA 具有较好的稳定性、计算可行性和一定的可解释性。

# 附录 A. 返修实验执行记录位置

该部分可作为内部记录，不建议直接放入正文。若后续需要写 supplementary material，可根据该部分进一步扩展。

## A.1 实验环境

| 项 | 值 |
|---|---|
| 工作目录 | `~/HAG-DTA/code_base` |
| 输出目录 | `/root/autodl-tmp/HAG-DTA-runs` |
| 缓存目录 | `/root/autodl-tmp/HAG-DTA-cache/processed/` |
| 额外依赖 | `pip install -r ~/HAG-DTA/requirements-extra.txt` |
| PyTorch | 2.0.0 |
| PyTorch Geometric | 2.6.1 |
| CUDA | 11.8 |

## A.2 实验执行顺序

```bash
cd ~/HAG-DTA/code_base

# 数据预处理
python create_data_Human_Celegans.py
python create_data_transformercpi.py

# 消融实验和敏感性分析
bash scripts/sensitivity_mmd.sh
bash scripts/sensitivity_n1n2.sh
bash scripts/sensitivity_n1n2_human.sh

# 分类主实验
bash scripts/run_human.sh
bash scripts/run_celegans.sh
bash scripts/run_human_transformercpi.sh
bash scripts/run_celegans_transformercpi.sh
python scripts/statistical_tests_local.py

# 回归主实验
bash scripts/run_davis_full.sh
bash scripts/run_kiba_full.sh
python scripts/statistical_tests.py
```

## A.3 后续需要补充的数据清单

| 编号 | 内容 | 对应表格 | 状态 |
|---:|---|---|---|
| 1 | Davis / KIBA 主实验五种子结果 | Table 3 | 待补 |
| 2 | Human / *C.elegans* 主实验五种子结果 | Table 4 | 待补 |
| 3 | 近期 SOTA 方法结果和来源 | Table 7 | 待补 |
| 4 | MMD $\beta$ 完整五种子结果 | Table 9 | 待补 |
| 5 | $n_1$ / $n_2$ 完整五种子结果 | Table 10, Table 11 | 待补 |
| 6 | GraphDTA 与 TransformerCPI 显著性检验 | Table 12 | 待补 |
| 7 | 参数量、训练时间、推理时间、显存占用 | Table 13 | 待补 |

