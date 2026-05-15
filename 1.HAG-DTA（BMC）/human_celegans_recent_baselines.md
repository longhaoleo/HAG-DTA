## Human 和 *C. elegans* DTI/CPI 数据集近期 baseline 检索整理

---

## 1. 最适合你当前表格的 baseline 来源

如果你的目标表格是：

| Model | Year | AUROC/AUC | AUPRC/AUPR | Accuracy | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|

最直接可用的是 **BiMA-DTI 2025** 论文中的 Table 1 和 Table 2。该论文在 Human 和 *C. elegans* 上采用 E1 random-split setting，并报告 10 random runs 的均值和标准差，指标包含 AUROC、AUPRC、Accuracy、F1-score 和 MCC。

原文链接：

- BiMA-DTI, BMC Biology, 2025: https://link.springer.com/article/10.1186/s12915-025-02407-4

### 1.1 Human 数据集，E1，10 random runs

| Model | Year | AUROC/AUC | AUPRC/AUPR | Accuracy | F1 | MCC | 适合程度 |
|---|---:|---:|---:|---:|---:|---:|---|
| MGNDTI | 2024 | 0.9855 ± 0.0031 | 0.9820 ± 0.0059 | 0.9481 ± 0.0056 | 0.9485 ± 0.0057 | 0.8951 ± 0.0111 | 适合，近年强 baseline |
| MolTrans | 2021 | 0.9799 ± 0.0028 | 0.9785 ± 0.0044 | 0.9418 ± 0.0099 | 0.9207 ± 0.0291 | 0.8823 ± 0.0196 | 适合，经典 baseline |
| TransformerCPI | 2020 | 0.9795 ± 0.0036 | 0.9745 ± 0.0052 | 0.9316 ± 0.0071 | 0.9223 ± 0.0092 | 0.8613 ± 0.0147 | 适合，经典 baseline |
| CPI-GNN | 2019 | 0.9329 ± 0.0085 | 0.9174 ± 0.0164 | 0.8899 ± 0.0084 | 0.8858 ± 0.0098 | 0.7798 ± 0.0170 | 很适合，指标不高，便于对比 |
| BACPI | 2023 | 0.9670 ± 0.0058 | 0.9608 ± 0.0087 | 0.9181 ± 0.0110 | 0.9070 ± 0.0130 | 0.8341 ± 0.0225 | 适合，指标适中 |
| CPGL | 2024 | 0.9674 ± 0.0052 | 0.9673 ± 0.0079 | 0.9092 ± 0.0123 | 0.9055 ± 0.0141 | 0.8191 ± 0.0238 | 适合，指标适中 |
| GIFDTI | 2024 | 0.9690 ± 0.0047 | 0.9645 ± 0.0084 | 0.9091 ± 0.0099 | 0.8967 ± 0.0120 | 0.8161 ± 0.0200 | 很适合，指标不高，便于对比 |
| FOTF-CPI | 2024 | 0.9834 ± 0.0024 | 0.9803 ± 0.0035 | 0.9413 ± 0.0074 | 0.9326 ± 0.0090 | 0.8811 ± 0.0149 | 适合，近年较强 baseline |
| DO-GMA | 2025 | 0.9891 ± 0.0029 | 0.9870 ± 0.0068 | 0.9541 ± 0.0052 | 0.9544 ± 0.0047 | 0.9098 ± 0.0101 | 指标很强，谨慎放主表 |
| LAM-DTI | 2025 | 0.9860 ± 0.0044 | 0.9854 ± 0.0024 | 0.9502 ± 0.0057 | 0.9497 ± 0.0056 | 0.9007 ± 0.0114 | 指标较强，可放主表 |
| BiMA-DTI | 2025 | 0.9860 ± 0.0039 | 0.9858 ± 0.0024 | 0.9523 ± 0.0053 | 0.9522 ± 0.0049 | 0.9043 ± 0.0101 | 指标很强，谨慎放主表 |





### 1.2 *C. elegans* 数据集，E1，10 random runs

| Model | Year | AUROC/AUC | AUPRC/AUPR | Accuracy | F1 | MCC | 适合程度 |
|---|---:|---:|---:|---:|---:|---:|---|
| MGNDTI | 2024 | 0.9912 ± 0.0022 | 0.9916 ± 0.0020 | 0.9682 ± 0.0051 | 0.9682 ± 0.0050 | 0.9365 ± 0.0102 | 适合，近年强 baseline |
| MolTrans | 2021 | 0.9918 ± 0.0024 | 0.9920 ± 0.0028 | 0.9670 ± 0.0032 | 0.9626 ± 0.0047 | 0.9342 ± 0.0064 | 适合，经典 baseline |
| TransformerCPI | 2020 | 0.9919 ± 0.0015 | 0.9916 ± 0.0018 | 0.9608 ± 0.0045 | 0.9608 ± 0.0044 | 0.9217 ± 0.0089 | 适合，经典 baseline |
| CPI-GNN | 2019 | 0.9536 ± 0.0079 | 0.9422 ± 0.0121 | 0.9146 ± 0.0100 | 0.9185 ± 0.0088 | 0.8292 ± 0.0201 | 很适合，指标不高，便于对比 |
| BACPI | 2023 | 0.9864 ± 0.0044 | 0.9869 ± 0.0039 | 0.9489 ± 0.0114 | 0.9487 ± 0.0117 | 0.8981 ± 0.0227 | 适合，指标适中 |
| CPGL | 2024 | 0.9757 ± 0.0034 | 0.9797 ± 0.0029 | 0.9265 ± 0.0117 | 0.9288 ± 0.0103 | 0.8538 ± 0.0224 | 很适合，指标不高，便于对比 |
| GIFDTI | 2024 | 0.9827 ± 0.0068 | 0.9843 ± 0.0056 | 0.9435 ± 0.0110 | 0.9435 ± 0.0116 | 0.8873 ± 0.0221 | 适合，指标适中 |
| FOTF-CPI | 2024 | 0.9919 ± 0.0030 | 0.9909 ± 0.0054 | 0.9663 ± 0.0051 | 0.9661 ± 0.0052 | 0.9328 ± 0.0101 | 适合，近年较强 baseline |
| DO-GMA | 2025 | 0.9926 ± 0.0026 | 0.9905 ± 0.0048 | 0.9716 ± 0.0033 | 0.9715 ± 0.0034 | 0.9645 ± 0.0056 | 指标很强，谨慎放主表 |
| LAM-DTI | 2025 | 0.9929 ± 0.0013 | 0.9914 ± 0.0013 | 0.9690 ± 0.0049 | 0.9690 ± 0.0051 | 0.9382 ± 0.0098 | 指标较强，可放主表 |
| BiMA-DTI | 2025 | 0.9933 ± 0.0021 | 0.9940 ± 0.0020 | 0.9724 ± 0.0041 | 0.9725 ± 0.0040 | 0.9448 ± 0.0079 | 指标很强，谨慎放主表 |

---

## 2. 有 AUC / AUPR / Precision / Recall / F1 的正式论文

### 2.1 MIFAM-DTI, 2024

原文链接：

- Frontiers in Genetics: https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1381997/full

该文使用 10-fold cross-validation，在 Human 和 *C. elegans* 上报告 AUC、AUPR、Precision、Recall 和 F1-score。

| Dataset | Model | Year | AUC | AUPR | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Human | MIFAM-DTI | 2024 | 0.9874 | 0.9876 | 0.9762 | 0.9145 | 0.9443 |
| *C. elegans* | MIFAM-DTI | 2024 | 0.9964 | 0.9956 | 0.9862 | 0.9625 | 0.9742 |

同时，该文表中还整理了若干旧 baseline，可用于补充表：

| Dataset | Model | Year | AUC | AUPR | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Human | DeepCPI | 2018/2019 | 0.9692 | 0.9399 | 0.9187 | 0.9210 | 0.9096 |
| Human | DeepConv-DTI | 2019 | 0.9738 | 0.9437 | 0.9295 | 0.9175 | 0.9204 |
| Human | MHSADTI | 2022 | 0.9822 | 0.9568 | 0.9472 | 0.9365 | 0.9346 |
| Human | AMMVF-DTI | 2023 | 0.9860 | - | 0.9760 | 0.9380 | - |
| *C. elegans* | DeepCPI | 2018/2019 | 0.9758 | 0.9571 | 0.9393 | 0.9271 | 0.9394 |
| *C. elegans* | DeepConv-DTI | 2019 | 0.9782 | 0.9711 | 0.9435 | 0.9423 | 0.9579 |
| *C. elegans* | MHSADTI | 2022 | 0.9838 | 0.9832 | 0.9465 | 0.9451 | 0.9763 |
| *C. elegans* | AMMVF-DTI | 2023 | 0.9900 | - | 0.9620 | 0.9600 | - |

使用建议：MIFAM-DTI 本身指标很高，特别是 *C. elegans* 上的 AUC/AUPR 很接近饱和。若想找“指标没那么高”的近期方法，AMMVF-DTI 可以放入主表，但它缺少 AUPR。

---

## 3. 有 AUC / Precision / Recall，但缺少 AUPR 的正式论文

### 3.1 只需要保存进表格已有的指标。如果表格中没有指标的话，就不需要额外新增。

表格格式与先前一致，最后在同级的 .bib 文件下加入这个对应的引用。, 2025

原文链接：

- PLOS One: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324146

该文在 Human 和 *C. elegans* 上报告 AUC、Precision 和 Recall，未在这两个数据集上报告 AUPR。

| Dataset | Model | Year | AUC | Precision | Recall | AUPR | 适合程度 |
|---|---|---:|---:|---:|---:|---:|---|
| Human | SSCPA-DTI | 2025 | 0.990 | 0.955 | 0.946 | - | 可放入 recent baseline，但缺 AUPR |
| *C. elegans* | SSCPA-DTI | 2025 | 0.992 | 0.971 | 0.955 | - | 可放入 recent baseline，但缺 AUPR |

该文表中还给出若干可对比模型：

| Dataset | Model | AUC | Precision | Recall |
|---|---|---:|---:|---:|
| Human | TransformerCPI | 0.973 | 0.916 | 0.925 |
| Human | CPI-GNN | 0.970 | 0.918 | 0.923 |
| Human | GanDTI | 0.983 | 0.933 | 0.960 |
| Human | IIFDTI | 0.984 | 0.946 | 0.947 |
| Human | CoaDTI-pro | 0.982 | 0.952 | 0.950 |
| Human | Wang's Method | 0.982 | 0.931 | 0.936 |
| *C. elegans* | RF | 0.902 | 0.821 | 0.844 |
| *C. elegans* | GCN | 0.975 | 0.921 | 0.927 |
| *C. elegans* | TransformerCPI | 0.988 | 0.952 | 0.953 |
| *C. elegans* | CPI-GNN | 0.978 | 0.938 | 0.929 |
| *C. elegans* | MHSADTI | 0.983 | 0.946 | 0.945 |
| *C. elegans* | CoaDTI-pro | 0.985 | 0.957 | 0.948 |
| *C. elegans* | Wang's Method | 0.987 | 0.949 | 0.948 |

使用建议：SSCPA-DTI 是 2025 年正式发表论文，AUC、Precision、Recall 齐全，且指标没有 BiMA-DTI / MIFAM-DTI 那么全面压制你的模型。缺点是没有 Human / *C. elegans* 的 AUPR。

### 3.2 HMSA-DTI, 2024

原文链接：

- Briefings in Bioinformatics: https://academic.oup.com/bib/article/25/4/bbae293/7699346

该文在 Human 和 *C. elegans* 上报告 AUC、Accuracy、Precision 和 Recall。由于 Human 和 *C. elegans* 被视为 balanced datasets，该文没有在这两个数据集上报告 AUPR。

| Dataset | Model | Year | AUC | Accuracy | Precision | Recall | AUPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Human | HMSA-DTI | 2024 | 0.9937 | 0.9589 | 0.9591 | 0.9528 | - |
| *C. elegans* | HMSA-DTI | 2024 | 0.9950 | 0.9667 | 0.9613 | 0.9746 | - |

使用建议：HMSA-DTI 是正式发表的 2024 年方法，但指标较高且缺少 AUPR。如果你的表格一定保留 AUPRC/AUPR 列，则不建议放主表，可放补充讨论。

---

## 4. 有 AUC / AUPR / Accuracy，但缺少 Precision、Recall、F1、MCC 的正式论文



### 4.2 MoleProLink-RL, 2025

原文链接：

- npj Digital Medicine: https://www.nature.com/articles/s41746-025-02158-0

该文在 Human 和 *C. elegans* 上报告 AUC/AUPR，但未提供 Precision、Recall、F1、MCC 的完整主表。

| Dataset | Model | Year | AUC | AUPR | Accuracy | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| Human | MoleProLink-RL | 2025 | 0.9734 | 0.9757 | - | - | - |
| *C. elegans* | MoleProLink-RL | 2025 | 0.9813 | 0.9892 | - | - | - |

使用建议：指标不算极高，适合放在“近期文献参考”中，但因为缺少 Accuracy/F1/MCC，不建议进入你当前格式的主表。

### 4.3 BEACON, 2024

原文链接：

- BMC Biology: https://link.springer.com/article/10.1186/s12915-024-02049-y

BEACON 在 Human 和 *C. elegans* 上使用 8:1:1 random split，并报告 AUC、AUPR、Precision、Recall 和 ACC。不过网页主文中直接可见的 Human / *C. elegans* 具体数值主要是 AUC。

| Dataset | Model | Year | AUC | AUPR | Precision | Recall | Accuracy | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Human | BEACON | 2024 | 0.989 | - | - | - | - | 原文说明 Table 1 有完整五项指标，但网页文本只直接显示 AUC |
| *C. elegans* | BEACON | 2024 | 0.996 | - | - | - | - | 指标很高，不建议主表强比较 |

使用建议：BEACON 是 2024 年 BMC Biology，适合在 Related Work 或 Discussion 中提到；如果要放入表格，建议下载原文 PDF 或补充材料确认完整数值。


---

## 8. BibTeX 候选条目

```bibtex
@article{shui2025bimadti,
  title={BiMA-DTI: a bidirectional Mamba-Attention hybrid framework for enhanced drug-target interaction prediction},
  author={Shui, Youyuan and Ge, Xuewen and Cao, Chen and Wang, Junjie and Hu, Jie and Liu, Yun},
  journal={BMC Biology},
  year={2025},
  volume={23},
  pages={309},
  doi={10.1186/s12915-025-02407-4}
}

@article{peng2024mgndti,
  title={MGNDTI: A Drug-Target Interaction Prediction Framework Based on Multimodal Representation Learning and the Gating Mechanism},
  author={Peng, Lihong and others},
  journal={Journal of Chemical Information and Modeling},
  year={2024},
  doi={10.1021/acs.jcim.4c00957}
}

@article{peng2025dogma,
  title={DO-GMA: An End-to-End Drug-Target Interaction Identification Framework with a Depthwise Overparameterized Convolutional Network and the Gated Multihead Attention Mechanism},
  author={Peng, Lihong and Mao, Jiale and Huang, Guohua and others},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  doi={10.1021/acs.jcim.4c02088}
}

@article{wei2025lamdti,
  title={Dynamic Prediction of Drug-Target Interactions via Cross-Modal Feature Mapping with Learnable Association Information},
  author={Wei, Z. and others},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  doi={10.1021/acs.jcim.4c02348}
}

@article{li2024mifamdti,
  title={MIFAM-DTI: a drug-target interactions predicting model based on multi-source information fusion and attention mechanism},
  author={Li, J. and others},
  journal={Frontiers in Genetics},
  year={2024},
  volume={15},
  doi={10.3389/fgene.2024.1381997}
}

@article{shi2025sscpadti,
  title={Prediction of drug-target interactions based on substructure subsequences and cross-public attention mechanism},
  author={Shi, Haikuo and Hu, Jing and Zhang, Xiaolong and others},
  journal={PLOS One},
  year={2025},
  volume={20},
  pages={e0324146},
  doi={10.1371/journal.pone.0324146}
}

@article{bian2024hmsadti,
  title={Hierarchical multimodal self-attention-based graph neural network for drug-target interaction prediction},
  author={Bian, J. and others},
  journal={Briefings in Bioinformatics},
  year={2024},
  volume={25},
  number={4},
  pages={bbae293},
  doi={10.1093/bib/bbae293}
}

@article{zhang2025cpimif,
  title={CPI-MIF: Compound-Protein Interaction Prediction with Multiview Information Fusion},
  author={Zhang, Yunuo and Wen, Bozhu and Li, Yaru and others},
  journal={ACS Omega},
  year={2025},
  volume={10},
  pages={30155--30166},
  doi={10.1021/acsomega.5c00113}
}

@article{tao2024beacon,
  title={Bridging chemical structure and conceptual knowledge enables accurate prediction of compound-protein interaction},
  author={Tao, Wen and Lin, Xuan and Liu, Yuansheng and Zeng, Li and Ma, Tengfei and others},
  journal={BMC Biology},
  year={2024},
  volume={22},
  pages={248},
  doi={10.1186/s12915-024-02049-y}
}

@article{xu2025moleprolink,
  title={MoleProLink-RL: geometric transport for domain-policy reinforcement learning in drug-target interaction prediction},
  author={Xu, D. and others},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02158-0}
}
```
