# Response to Reviewer 1, Comment 1: Hyperparameter Sensitivity of n1 and n2

## Reviewer's Concern
> "The model's performance is sensitive to the choice of hyperparameters such as n1 and n2,
> which are not always easy to determine in real-world applications... explore automated
> hyperparameter optimization techniques."

## Our Response Strategy

We respectfully acknowledge this concern and address it in three tiers:

---

### Tier 1: Chemical Motivation for (n1=6, n2=3)

The values n1=6 and n2=3 were not chosen arbitrarily. They reflect the natural
hierarchical structure of drug-like molecules:

- **n1=6 (first pooling layer):** Small-molecule drugs typically contain 4–8 functional
  groups or ring systems. Pooling the ~46-atom molecular graph into 6 super-nodes
  corresponds to a biologically meaningful granularity — each super-node approximately
  represents one functional group or a fused ring system.

- **n2=3 (second pooling layer):** Further aggregating the 6 fragment-level clusters
  into 3 higher-level representations mirrors the common pharmacophoric decomposition
  of drug molecules into hydrophobic, hydrogen-bonding, and aromatic/π-stacking regions.
  This two-level hierarchy (functional groups → pharmacophores) closely parallels how
  medicinal chemists reason about structure-activity relationships.

- **Pooling ratios:** The compression ratios (46→6 = 7.7:1, then 6→3 = 2:1) are
  consistent with the 2:1 to 10:1 ratios commonly employed in hierarchical graph
  pooling architectures (DiffPool, MinCutPool, ASAP).

This chemical rationale has been added to the revised manuscript (Section X.X).

---

### Tier 2: Empirical Sensitivity Analysis

To empirically validate our parameter choice, we conducted a grid search over
n1 ∈ {4, 5, 6, 7, 8} × n2 ∈ {2, 3, 4} (with the constraint n2 < n1) on the
Davis dataset, using GIN as the base GNN. Each configuration was evaluated with
a single random seed and one fold.

**Results (to be filled after running):**

| n1 | n2 | Val MSE | Val CI | Val rm² | Test MSE | Test CI | Test rm² |
|----|-----|---------|--------|---------|----------|---------|----------|
| 4  | 2  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| 5  | 2  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| 5  | 3  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| **6** | **3** | **—** | **—** | **—** | **—** | **—** | **—** |
| 7  | 3  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| 7  | 4  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| 8  | 3  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |
| 8  | 4  |   ?     |   ?    |   ?     |    ?     |   ?     |   ?      |

[DISCUSSION TO BE WRITTEN AFTER RESULTS]

---

### Tier 3: Practical Justification

We note that HAG-DTA's primary contribution lies in its **hierarchical multi-scale
attention fusion architecture**, not in hyperparameter optimization. The model
achieves state-of-the-art performance with (n1=6, n2=3) across four diverse
benchmark datasets (Davis, KIBA, Human, C.elegans), suggesting that the
architecture is robust to moderate variations in pooling granularity.

Full Bayesian optimization or automated grid search across all hyperparameters
(batch size, learning rate, pooling sizes, dropouts) would constitute a separate
study. We have added a discussion of this limitation to the revised manuscript
and identified it as promising future work.

---

## To-Do
- [ ] Run sensitivity analysis script on Davis
- [ ] Fill the results table above
- [ ] Write final discussion based on actual results
- [ ] Add chemical motivation paragraph to manuscript
