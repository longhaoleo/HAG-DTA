# HAG-DTA 引用文献 BibTeX 汇总

---

## 一、DTA 模型及 GNN 基础方法

| BibTeX key | LaTeX cite | 文献名称 | 年份 | DOI |
|---|---|---|---|---|
| `KronRLS` | `\cite{KronRLS}` | Toward more realistic drug–target interaction predictions | 2015 | 10.1093/bib/bbu010 |
| `SimBoost` | `\cite{SimBoost}` | SimBoost: a read-across approach for predicting drug–target binding affinities using gradient boosting machines | 2017 | 10.1186/s13321-017-0209-z |
| `DeepDTA` | `\cite{DeepDTA}` | DeepDTA: deep drug–target binding affinity prediction | 2018 | 10.1093/bioinformatics/bty593 |
| `WideDTA` | `\cite{WideDTA}` | WideDTA: prediction of drug–target binding affinity | 2019 | arXiv:1902.04166 |
| `AttentionDTA` | `\cite{AttentionDTA}` | AttentionDTA: drug–target binding affinity prediction by sequence-based deep learning with attention mechanism | 2023 | 10.1109/TCBB.2022.3170365 |
| `TransformerCPI` | `\cite{TransformerCPI}` | TransformerCPI: improving compound–protein interaction prediction... | 2020 | 10.1093/bioinformatics/btaa524 |
| `MolTrans` | `\cite{MolTrans}` | MolTrans: Molecular Interaction Transformer for drug–target interaction prediction | 2021 | 10.1093/bioinformatics/btaa880 |
| `GraphDTA` | `\cite{GraphDTA}` | GraphDTA: predicting drug–target binding affinity with graph neural networks | 2021 | 10.1093/bioinformatics/btaa921 |
| `MGraphDTA` | `\cite{MGraphDTA}` | MGraphDTA: deep multiscale graph neural network... | 2022 | 10.1039/d1sc05180f |
| `DGraphDTA` | `\cite{DGraphDTA}` | Drug–target affinity prediction using graph neural network and contact maps | 2020 | 10.1039/d0ra02297g |
| `AttentionMGTDTA` | `\cite{AttentionMGTDTA}` | AttentionMGT-DTA: A multi-modal drug-target affinity prediction... | 2024 | 10.1016/j.neunet.2023.11.018 |
| `GPCNDTA` | `\cite{GPCNDTA}` | GPCNDTA: Prediction of drug-target binding affinity... | 2023 | 10.1016/j.compbiomed.2023.107512 |
| `GEFormerDTA` | `\cite{GEFormerDTA}` | GEFormerDTA: drug target affinity prediction based on transformer graph... | 2024 | 10.1038/s41598-024-57879-1 |
| `GNPDTA` | `\cite{GNPDTA}` | Graph neural pre-training based drug-target affinity prediction | 2024 | 10.3389/fgene.2024.1452339 |
| `GLCNDTA` | `\cite{GLCNDTA}` | Drug–target affinity prediction with extended graph learning-convolutional networks | 2024 | 10.1186/s12859-024-05698-6 |
| `CSCoDTA` | `\cite{CSCoDTA}` | Predicting drug–target binding affinity with cross-scale graph contrastive learning | 2024 | 10.1093/bib/bbad516 |
| `MDCTDTA` | `\cite{MDCTDTA}` | Drug–target binding affinity prediction model based on multi-scale diffusion... | 2024 | 10.1016/j.eswa.2024.124647 |
| `WPGraphDTA` | `\cite{WPGraphDTA}` | Drug-target binding affinity prediction based on power graph and Word2Vec | 2025 | 10.1186/s12920-024-02073-5 |
| `MEGDTA` | `\cite{MEGDTA}` | MEGDTA: multi-modal drug-target affinity prediction... | 2025 | 10.1186/s12864-025-11943-w |
| `MONN` | `\cite{MONN}` | MONN: A Multi-objective Neural Network... | 2020 | 10.1016/j.cels.2020.03.002 |
| `GCNs` | `\cite{GCNs}` | Semi-Supervised Classification with Graph Convolutional Networks | 2017 | arXiv:1609.02907 |
| `GINs` | `\cite{GINs}` | How Powerful are Graph Neural Networks? | 2019 | arXiv:1810.00826 |
| `GATs` | `\cite{GATs}` | Graph Attention Networks | 2018 | arXiv:1710.10903 |
| `GraphSAGE` | `\cite{GraphSAGE}` | Inductive Representation Learning on Large Graphs | 2017 | arXiv:1706.02216 |
| `DiffPool` | `\cite{DiffPool}` | Hierarchical Graph Representation Learning with Differentiable Pooling | 2018 | arXiv:1806.08804 |
| `DPSP` | `\cite{DPSP}` | DPSP: a multimodal deep learning framework... | 2023 | 10.1093/bioadv/vbad110 |
| `NNPS` | `\cite{NNPS}` | A neural network-based method for polypharmacy side effects prediction | 2021 | 10.1186/s12859-021-04298-y |

---

## 二、审稿人要求补充引用 (PMID)

| Key | LaTeX cite | 文献名称 | 年份 | DOI |
|---|---|---|---|---|
| `VGAELDA` | `\cite{VGAELDA}` | A representation learning model based on variational inference and graph autoencoder for predicting lncRNA-disease associations | 2021 | 10.1186/s12859-021-04073-z |
| `DAttProt` | `\cite{DAttProt}` | An Interpretable Double-Scale Attention Model for Enzyme Protein Class Prediction Based on Transformer Encoders and Multi-Scale Convolutions | 2022 | 10.3389/fgene.2022.885627 |
| `NIMGSA` | `\cite{NIMGSA}` | Predicting miRNA-Disease Association Based on Neural Inductive Matrix Completion with Graph Autoencoders and Self-Attention Mechanism | 2022 | 10.3390/biom12010064 |
| `TLCrys` | `\cite{TLCrys}` | TLCrys: Transfer Learning Based Method for Protein Crystallization Prediction | 2022 | 10.3390/ijms23020972 |

---

## BibTeX 条目

### KronRLS

```bibtex
@article{KronRLS,
  author = {Pahikkala, Tapio and Airola, Antti and Pietilä, Sami and Shakyawar, Sushil and Szwajda, Agnieszka and Tang, Jing and Aittokallio, Tero},
  title = {Toward more realistic drug–target interaction predictions},
  journal = {Briefings in Bioinformatics},
  volume = {16},
  number = {2},
  pages = {325--337},
  year = {2015},
  doi = {10.1093/bib/bbu010}
}
```

### SimBoost
```bibtex
@article{SimBoost,
  author = {He, Tong and Heidemeyer, Marten and Ban, Fuqiang and Cherkasov, Artem and Ester, Martin},
  title = {{SimBoost}: a read-across approach for predicting drug–target binding affinities using gradient boosting machines},
  journal = {Journal of Cheminformatics},
  volume = {9},
  number = {1},
  year = {2017},
  doi = {10.1186/s13321-017-0209-z}
}
```

### DeepDTA
```bibtex
@article{DeepDTA,
  author = {{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  title = {{DeepDTA}: deep drug–target binding affinity prediction},
  journal = {Bioinformatics},
  volume = {34},
  number = {17},
  pages = {i821--i829},
  year = {2018},
  doi = {10.1093/bioinformatics/bty593}
}
```

### WideDTA
```bibtex
@article{WideDTA,
  author = {{\"O}zt{\"u}rk, Hakime and Ozkirimli, Elif and {\"O}zg{\"u}r, Arzucan},
  title = {{WideDTA}: prediction of drug–target binding affinity},
  year = {2019},
  eprint = {1902.04166},
  archivePrefix = {arXiv},
  note = {arXiv:1902.04166}
}
```

### AttentionDTA
```bibtex
@article{AttentionDTA,
  author = {Zhao, Qihang and Zhao, Hong and Zheng, Kai and Wang, Jianxin},
  title = {{AttentionDTA}: drug–target binding affinity prediction by sequence-based deep learning with attention mechanism},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  volume = {20},
  number = {2},
  pages = {852--863},
  year = {2023},
  doi = {10.1109/TCBB.2022.3170365}
}
```

### TransformerCPI
```bibtex
@article{TransformerCPI,
  author = {Chen, Lifan and Tan, Xiaoqin and Wang, Dingyan and Zhong, Feisheng and Liu, Xiaohong and Yang, Tianbiao and Luo, Xiaomin and Chen, Kaixian and Jiang, Hualiang and Zheng, Mingyue},
  title = {{TransformerCPI}: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments},
  journal = {Bioinformatics},
  volume = {36},
  number = {16},
  pages = {4406--4414},
  year = {2020},
  doi = {10.1093/bioinformatics/btaa524}
}
```

### MolTrans
```bibtex
@article{MolTrans,
  author = {Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
  title = {{MolTrans}: Molecular Interaction Transformer for drug–target interaction prediction},
  journal = {Bioinformatics},
  volume = {37},
  number = {6},
  pages = {830--836},
  year = {2021},
  doi = {10.1093/bioinformatics/btaa880}
}
```

### GraphDTA
```bibtex
@article{GraphDTA,
  author = {Nguyen, Thin and Le, Hang and Quinn, Thomas P and Nguyen, Tri and Le, Thuc Duy and Venkatesh, Svetha},
  title = {{GraphDTA}: predicting drug–target binding affinity with graph neural networks},
  journal = {Bioinformatics},
  volume = {37},
  number = {8},
  pages = {1140--1147},
  year = {2021},
  doi = {10.1093/bioinformatics/btaa921}
}
```

### MGraphDTA
```bibtex
@article{MGraphDTA,
  author = {Yang, Ziduo and Zhong, Weihe and Zhao, Lu and Yu-Chian Chen, Calvin},
  title = {{MGraphDTA}: deep multiscale graph neural network for explainable drug–target binding affinity prediction},
  journal = {Chemical Science},
  volume = {13},
  number = {3},
  pages = {816--833},
  year = {2022},
  doi = {10.1039/d1sc05180f}
}
```

### DGraphDTA
```bibtex
@article{DGraphDTA,
  author = {Jiang, Mingjian and Li, Zhen and Zhang, Shugang and Wang, Shuang and Wang, Xiaofeng and Yuan, Qing and Wei, Zhiqiang},
  title = {Drug–target affinity prediction using graph neural network and contact maps},
  journal = {RSC Advances},
  volume = {10},
  number = {35},
  pages = {20701--20712},
  year = {2020},
  doi = {10.1039/d0ra02297g}
}
```

### AttentionMGTDTA
```bibtex
@article{AttentionMGTDTA,
  author = {Wu, Hongjie and Liu, Junkai and Jiang, Tengsheng and Zou, Quan and Qi, Shujie and Cui, Zhiming and Tiwari, Prayag and Ding, Yijie},
  title = {{AttentionMGT-DTA}: A multi-modal drug-target affinity prediction using graph transformer and attention mechanism},
  journal = {Neural Networks},
  volume = {169},
  pages = {623--636},
  year = {2024},
  doi = {10.1016/j.neunet.2023.11.018}
}
```

### GPCNDTA
```bibtex
@article{GPCNDTA,
  author = {Zhang, Li and Wang, Chun-Chun and Zhang, Yong and Chen, Xing},
  title = {{GPCNDTA}: Prediction of drug-target binding affinity through cross-attention networks augmented with graph features and pharmacophores},
  journal = {Computers in Biology and Medicine},
  volume = {166},
  pages = {107512},
  year = {2023},
  doi = {10.1016/j.compbiomed.2023.107512}
}
```

### GEFormerDTA
```bibtex
@article{GEFormerDTA,
  author = {Liu, Youzhi and Xing, Linlin and Zhang, Longbo and Cai, Hongzhen and Guo, Maozu},
  title = {{GEFormerDTA}: drug target affinity prediction based on transformer graph for early fusion},
  journal = {Scientific Reports},
  volume = {14},
  number = {1},
  year = {2024},
  doi = {10.1038/s41598-024-57879-1}
}
```

### GNPDTA
```bibtex
@article{GNPDTA,
  author = {Ye, Qing and Sun, Yaxin},
  title = {Graph neural pre-training based drug-target affinity prediction},
  journal = {Frontiers in Genetics},
  volume = {15},
  year = {2024},
  doi = {10.3389/fgene.2024.1452339}
}
```

### GLCNDTA
```bibtex
@article{GLCNDTA,
  author = {Qi, Haiou and Yu, Ting and Yu, Wenwen and Liu, Chenxi},
  title = {Drug–target affinity prediction with extended graph learning-convolutional networks},
  journal = {BMC Bioinformatics},
  volume = {25},
  number = {1},
  year = {2024},
  doi = {10.1186/s12859-024-05698-6}
}
```

### CSCoDTA
```bibtex
@article{CSCoDTA,
  author = {Wang, Jingru and Xiao, Yihang and Shang, Xuequn and Peng, Jiajie},
  title = {Predicting drug–target binding affinity with cross-scale graph contrastive learning},
  journal = {Briefings in Bioinformatics},
  volume = {25},
  number = {1},
  year = {2024},
  doi = {10.1093/bib/bbad516}
}
```

### MDCTDTA
```bibtex
@article{MDCTDTA,
  author = {Zhu, Zhiqin and Zheng, Xin and Qi, Guanqiu and Gong, Yifei and Li, Yuanyuan and Mazur, Neal and Cong, Baisen and Gao, Xinbo},
  title = {Drug–target binding affinity prediction model based on multi-scale diffusion and interactive learning},
  journal = {Expert Systems with Applications},
  volume = {255},
  pages = {124647},
  year = {2024},
  doi = {10.1016/j.eswa.2024.124647}
}
```

### WPGraphDTA
```bibtex
@article{WPGraphDTA,
  author = {Hu, Jing and Hu, Shuo and Xia, Minghao and Zheng, Kangxing and Zhang, Xiaolong},
  title = {Drug-target binding affinity prediction based on power graph and {Word2Vec}},
  journal = {BMC Medical Genomics},
  volume = {18},
  number = {S1},
  year = {2025},
  doi = {10.1186/s12920-024-02073-5}
}
```

### MEGDTA
```bibtex
@article{MEGDTA,
  author = {Hou, Zhanwei and Li, Yijun and Zhai, Haixia and Luo, Junwei and Ding, Yulian and Pan, Yi},
  title = {{MEGDTA}: multi-modal drug-target affinity prediction based on protein three-dimensional structure and ensemble graph neural network},
  journal = {BMC Genomics},
  volume = {26},
  number = {1},
  year = {2025},
  doi = {10.1186/s12864-025-11943-w}
}
```

### MONN
```bibtex
@article{MONN,
  author = {Li, Shuya and Wan, Fangping and Shu, Hantao and Jiang, Tao and Zhao, Dan and Zeng, Jianyang},
  title = {{MONN}: A Multi-objective Neural Network for Predicting Compound-Protein Interactions and Affinities},
  journal = {Cell Systems},
  volume = {10},
  number = {4},
  pages = {308--322.e11},
  year = {2020},
  doi = {10.1016/j.cels.2020.03.002}
}
```

### GCN
```bibtex
@inproceedings{GCN,
  author = {Kipf, Thomas N. and Welling, Max},
  title = {Semi-Supervised Classification with Graph Convolutional Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2017},
  eprint = {1609.02907},
  archivePrefix = {arXiv},
}
```

### GIN
```bibtex
@inproceedings{GIN,
  author = {Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  title = {How Powerful are Graph Neural Networks?},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2019},
  eprint = {1810.00826},
  archivePrefix = {arXiv},
}
```

### GAT
```bibtex
@inproceedings{GAT,
  author = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  title = {Graph Attention Networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2018},
  eprint = {1710.10903},
  archivePrefix = {arXiv},
}
```

### GraphSAGE
```bibtex
@inproceedings{GraphSAGE,
  author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title = {Inductive Representation Learning on Large Graphs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017},
  eprint = {1706.02216},
  archivePrefix = {arXiv},
}
```

### DiffPool
```bibtex
@inproceedings{DiffPool,
  author = {Ying, Rex and You, Jiaxuan and Morris, Christopher and Ren, Xiang and Hamilton, William L. and Leskovec, Jure},
  title = {Hierarchical Graph Representation Learning with Differentiable Pooling},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2018},
  eprint = {1806.08804},
  archivePrefix = {arXiv},
}
```

### DPSP
```bibtex
@article{DPSP,
  author = {Masumshah, Raziyeh and Eslahchi, Changiz},
  title = {{DPSP}: a multimodal deep learning framework for polypharmacy side effects prediction},
  journal = {Bioinformatics Advances},
  volume = {3},
  number = {1},
  year = {2023},
  doi = {10.1093/bioadv/vbad110}
}
```

### NNPS
```bibtex
@article{NNPS,
  author = {Masumshah, Raziyeh and Aghdam, Rosa and Eslahchi, Changiz},
  title = {A neural network-based method for polypharmacy side effects prediction},
  journal = {BMC Bioinformatics},
  volume = {22},
  number = {1},
  year = {2021},
  doi = {10.1186/s12859-021-04298-y}
}
```

### VGAELDA
```bibtex
@article{VGAELDA,
  author = {Shi, Zhan and Zhang, Heng and Jin, Cheng and Quan, Xiongwen and Yin, Yanbin},
  title = {A representation learning model based on variational inference and graph autoencoder for predicting {lncRNA}-disease associations},
  journal = {BMC Bioinformatics},
  volume = {22},
  number = {1},
  pages = {136},
  year = {2021},
  doi = {10.1186/s12859-021-04073-z}
}
```

### DAttProt
```bibtex
@article{DAttProt,
  author = {Lin, Kang and Quan, Xiongwen and Jin, Cheng and Shi, Zhan and Yang, Jing},
  title = {An Interpretable Double-Scale Attention Model for Enzyme Protein Class Prediction Based on Transformer Encoders and Multi-Scale Convolutions},
  journal = {Frontiers in Genetics},
  volume = {13},
  pages = {885627},
  year = {2022},
  doi = {10.3389/fgene.2022.885627}
}
```

### NIMGSA
```bibtex
@article{NIMGSA,
  author = {Jin, Cheng and Shi, Zhan and Lin, Kang and Zhang, Heng},
  title = {Predicting {miRNA}-Disease Association Based on Neural Inductive Matrix Completion with Graph Autoencoders and Self-Attention Mechanism},
  journal = {Biomolecules},
  volume = {12},
  number = {1},
  pages = {64},
  year = {2022},
  doi = {10.3390/biom12010064}
}
```

### TLCrys
```bibtex
@article{TLCrys,
  author = {Jin, Cheng and Shi, Zhan and Kang, Chun and Lin, Kang and Zhang, Heng},
  title = {{TLCrys}: Transfer Learning Based Method for Protein Crystallization Prediction},
  journal = {International Journal of Molecular Sciences},
  volume = {23},
  number = {2},
  pages = {972},
  year = {2022},
  doi = {10.3390/ijms23020972}
}
```
