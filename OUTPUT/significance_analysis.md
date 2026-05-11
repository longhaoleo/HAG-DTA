# Significance Analysis From Logs

KIBA is not included because it is still running. This report covers Davis, Human, and C.elegans.

Primary significance test: two-sided Welch's t-test over shared seeds. A paired t-test is also exported in the CSV because each comparison uses the same seed list.

## Extracted Per-Seed Results

### Davis
| dataset   | model        |   seed |   best_epoch |    mse |     ci |    rm2 |
|:----------|:-------------|-------:|-------------:|-------:|-------:|-------:|
| Davis     | HAG-DTA GIN  |    100 |          400 | 0.2024 | 0.9067 | 0.7305 |
| Davis     | GraphDTA GIN |    100 |          500 | 0.2306 | 0.8881 | 0.7080 |
| Davis     | HAG-DTA GIN  |   1000 |          891 | 0.1964 | 0.9104 | 0.7401 |
| Davis     | GraphDTA GIN |   1000 |          974 | 0.2281 | 0.8937 | 0.7028 |
| Davis     | HAG-DTA GIN  |   2000 |          661 | 0.2013 | 0.9028 | 0.7292 |
| Davis     | GraphDTA GIN |   2000 |          493 | 0.2270 | 0.8913 | 0.7041 |
| Davis     | HAG-DTA GIN  |   3000 |          819 | 0.1989 | 0.9086 | 0.7338 |
| Davis     | GraphDTA GIN |   3000 |          823 | 0.2263 | 0.8917 | 0.7094 |
| Davis     | HAG-DTA GIN  |   4000 |          762 | 0.1977 | 0.9049 | 0.7525 |
| Davis     | GraphDTA GIN |   4000 |          703 | 0.2268 | 0.8933 | 0.7148 |

### Human
| dataset   | model          |   seed |   best_epoch |   AUROC |   AUPRC |   Precision |   Recall |
|:----------|:---------------|-------:|-------------:|--------:|--------:|------------:|---------:|
| Human     | HAG-DTA GIN    |    100 |          330 |  0.9717 |  0.9692 |      0.9223 |   0.9355 |
| Human     | HAG-DTA GIN    |   1000 |          201 |  0.9752 |  0.9667 |      0.9420 |   0.9319 |
| Human     | HAG-DTA GIN    |   2000 |          411 |  0.9797 |  0.9788 |      0.9353 |   0.9319 |
| Human     | HAG-DTA GIN    |   3000 |          266 |  0.9792 |  0.9756 |      0.9288 |   0.9355 |
| Human     | HAG-DTA GIN    |   4000 |          345 |  0.9758 |  0.9759 |      0.9321 |   0.9355 |
| Human     | TransformerCPI |    100 |           14 |  0.9734 |  0.9723 |      0.8976 |   0.9313 |
| Human     | TransformerCPI |   1000 |           15 |  0.9696 |  0.9690 |      0.8795 |   0.9125 |
| Human     | TransformerCPI |   2000 |           10 |  0.9750 |  0.9722 |      0.9211 |   0.9125 |
| Human     | TransformerCPI |   3000 |           14 |  0.9714 |  0.9692 |      0.8902 |   0.9375 |
| Human     | TransformerCPI |   4000 |           11 |  0.9700 |  0.9654 |      0.9027 |   0.9281 |

### C.elegans
| dataset   | model          |   seed |   best_epoch |   AUROC |   AUPRC |   Precision |   Recall |
|:----------|:---------------|-------:|-------------:|--------:|--------:|------------:|---------:|
| C.elegans | HAG-DTA GIN    |    100 |          218 |  0.9888 |  0.9890 |      0.9649 |   0.9450 |
| C.elegans | HAG-DTA GIN    |   1000 |           82 |  0.9890 |  0.9889 |      0.9172 |   0.9519 |
| C.elegans | HAG-DTA GIN    |   2000 |           85 |  0.9893 |  0.9883 |      0.9647 |   0.9381 |
| C.elegans | HAG-DTA GIN    |   3000 |          236 |  0.9905 |  0.9900 |      0.9456 |   0.9553 |
| C.elegans | HAG-DTA GIN    |   4000 |          159 |  0.9912 |  0.9915 |      0.9402 |   0.9725 |
| C.elegans | TransformerCPI |    100 |            3 |  0.9831 |  0.9819 |      0.9311 |   0.9311 |
| C.elegans | TransformerCPI |   1000 |           10 |  0.9843 |  0.9802 |      0.9478 |   0.9504 |
| C.elegans | TransformerCPI |   2000 |            9 |  0.9844 |  0.9826 |      0.9156 |   0.9559 |
| C.elegans | TransformerCPI |   3000 |            4 |  0.9852 |  0.9854 |      0.8171 |   0.9725 |
| C.elegans | TransformerCPI |   4000 |           11 |  0.9862 |  0.9853 |      0.9688 |   0.9421 |

## Significance Summary

### Davis
| Metric   | HAG-DTA         | Baseline        |   Delta | Welch p   | Significant   |
|:---------|:----------------|:----------------|--------:|:----------|:--------------|
| mse      | 0.1993 ± 0.0025 | 0.2278 ± 0.0017 | -0.0284 | <0.001    | Yes           |
| ci       | 0.9067 ± 0.0030 | 0.8916 ± 0.0022 |  0.0151 | <0.001    | Yes           |
| rm2      | 0.7372 ± 0.0095 | 0.7078 ± 0.0048 |  0.0294 | <0.001    | Yes           |

### Human
| Metric    | HAG-DTA         | Baseline        |   Delta |   Welch p | Significant   |
|:----------|:----------------|:----------------|--------:|----------:|:--------------|
| AUROC     | 0.9763 ± 0.0033 | 0.9719 ± 0.0023 |  0.0044 |    0.0408 | Yes           |
| AUPRC     | 0.9732 ± 0.0051 | 0.9696 ± 0.0028 |  0.0036 |    0.2106 | No            |
| Precision | 0.9321 ± 0.0073 | 0.8982 ± 0.0155 |  0.0339 |    0.005  | Yes           |
| Recall    | 0.9341 ± 0.0020 | 0.9244 ± 0.0114 |  0.0097 |    0.1296 | No            |

### C.elegans
| Metric    | HAG-DTA         | Baseline        |   Delta | Welch p   | Significant   |
|:----------|:----------------|:----------------|--------:|:----------|:--------------|
| AUROC     | 0.9898 ± 0.0010 | 0.9846 ± 0.0012 |  0.0051 | <0.001    | Yes           |
| AUPRC     | 0.9895 ± 0.0013 | 0.9831 ± 0.0022 |  0.0065 | 0.0012    | Yes           |
| Precision | 0.9465 ± 0.0198 | 0.9161 ± 0.0588 |  0.0304 | 0.3233    | No            |
| Recall    | 0.9526 ± 0.0130 | 0.9504 ± 0.0155 |  0.0022 | 0.8172    | No            |

## Notes

- Davis compares HAG-DTA GIN against GraphDTA GIN using MSE, CI, and rm2. For MSE, lower is better; for CI and rm2, higher is better.
- Human and C.elegans compare HAG-DTA GIN against TransformerCPI using AUROC, AUPRC, Precision, and Recall.
- Accuracy, F1, and MCC are not included here because the existing logs do not contain enough prediction-level information to reconstruct them.
- The extracted CSV and test summary CSV are generated next to this Markdown file.
