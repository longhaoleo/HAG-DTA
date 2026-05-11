#!/usr/bin/env python3
"""
Extract per-seed results from OUTPUT/logs and run significance tests.

Current completed comparisons:
- Davis: HAG-DTA GIN vs GraphDTA GIN
- Human: HAG-DTA GIN vs TransformerCPI
- C.elegans: HAG-DTA GIN vs TransformerCPI

KIBA is intentionally omitted until its logs are complete.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "OUTPUT"
LOG_DIR = OUTPUT_DIR / "logs"

SEEDS = [100, 1000, 2000, 3000, 4000]


@dataclass(frozen=True)
class Comparison:
    dataset: str
    task: str
    left_model: str
    right_model: str
    metrics: list[str]
    higher_is_better: set[str]


COMPARISONS = [
    Comparison(
        dataset="Davis",
        task="regression",
        left_model="HAG-DTA GIN",
        right_model="GraphDTA GIN",
        metrics=["mse", "ci", "rm2"],
        higher_is_better={"ci", "rm2"},
    ),
    Comparison(
        dataset="Human",
        task="classification",
        left_model="HAG-DTA GIN",
        right_model="TransformerCPI",
        metrics=["AUROC", "AUPRC", "Precision", "Recall"],
        higher_is_better={"AUROC", "AUPRC", "Precision", "Recall"},
    ),
    Comparison(
        dataset="C.elegans",
        task="classification",
        left_model="HAG-DTA GIN",
        right_model="TransformerCPI",
        metrics=["AUROC", "AUPRC", "Precision", "Recall"],
        higher_is_better={"AUROC", "AUPRC", "Precision", "Recall"},
    ),
]


def read_log(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing log: {path}")
    return path.read_text(errors="replace")


def parse_regression_final(text: str) -> dict[str, float]:
    pattern = re.compile(
        r"final test \| epoch\s+(?P<epoch>\d+) \| "
        r"mse=(?P<mse>[0-9.]+) ci=(?P<ci>[0-9.]+) rm2=(?P<rm2>[0-9.]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("No regression final-test line found")
    m = matches[-1]
    return {
        "best_epoch": int(m.group("epoch")),
        "mse": float(m.group("mse")),
        "ci": float(m.group("ci")),
        "rm2": float(m.group("rm2")),
    }


def parse_hag_classification_best(text: str) -> dict[str, float]:
    pattern = re.compile(
        r"\[FIND BEST!\] epoch (?P<epoch>\d+) \| "
        r"val AUROC=(?P<val_auroc>[0-9.]+) AUPRC=(?P<val_auprc>[0-9.]+)"
        r" Precision=(?P<val_precision>[0-9.]+) Recall=(?P<val_recall>[0-9.]+)\|\s*"
        r"test AUROC=(?P<AUROC>[0-9.]+) AUPRC=(?P<AUPRC>[0-9.]+) "
        r"Precision=(?P<Precision>[0-9.]+) Recall=(?P<Recall>[0-9.]+)",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("No HAG-DTA classification best-test block found")
    m = matches[-1]
    return {
        "best_epoch": int(m.group("epoch")),
        "AUROC": float(m.group("AUROC")),
        "AUPRC": float(m.group("AUPRC")),
        "Precision": float(m.group("Precision")),
        "Recall": float(m.group("Recall")),
    }


def parse_transformercpi_log(text: str, seed: int) -> dict[str, float]:
    seed_header = f"running on {{dataset}}, seed={seed}"
    sections = re.split(r"\nrunning on (?P<dataset>Human|Celegans), seed=(?P<seed>\d+)\n", text)
    for idx in range(1, len(sections), 3):
        dataset = sections[idx]
        section_seed = int(sections[idx + 1])
        body = sections[idx + 2]
        if section_seed != seed:
            continue
        pattern = re.compile(
            r"epoch (?P<epoch>\d+) \| loss=[0-9.]+ \| "
            r"dev AUROC=(?P<dev_auroc>[0-9.]+) \| "
            r"test AUROC=(?P<AUROC>[0-9.]+) AUPRC=(?P<AUPRC>[0-9.]+) "
            r"Precision=(?P<Precision>[0-9.]+) Recall=(?P<Recall>[0-9.]+)"
        )
        matches = list(pattern.finditer(body))
        if not matches:
            raise ValueError(f"No TransformerCPI metric line found for seed {seed}")
        m = matches[-1]
        return {
            "best_epoch": int(m.group("epoch")),
            "AUROC": float(m.group("AUROC")),
            "AUPRC": float(m.group("AUPRC")),
            "Precision": float(m.group("Precision")),
            "Recall": float(m.group("Recall")),
        }
    raise ValueError(seed_header.replace("{dataset}", "Human/Celegans") + " not found")


def collect_seed_results() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for prefix, model in [("davis", "HAG-DTA GIN"), ("davis_graphdta", "GraphDTA GIN")]:
            metrics = parse_regression_final(read_log(LOG_DIR / f"{prefix}_s{seed}.log"))
            rows.append({"dataset": "Davis", "task": "regression", "model": model, "seed": seed, **metrics})

        for dataset, prefix in [("Human", "human"), ("C.elegans", "celegans")]:
            metrics = parse_hag_classification_best(read_log(LOG_DIR / f"{prefix}_s{seed}.log"))
            rows.append({"dataset": dataset, "task": "classification", "model": "HAG-DTA GIN", "seed": seed, **metrics})

    for dataset, log_name in [("Human", "human_transformercpi.log"), ("C.elegans", "celegans_transformercpi.log")]:
        text = read_log(LOG_DIR / log_name)
        for seed in SEEDS:
            metrics = parse_transformercpi_log(text, seed)
            rows.append({"dataset": dataset, "task": "classification", "model": "TransformerCPI", "seed": seed, **metrics})

    return pd.DataFrame(rows)


def format_mean_std(values: pd.Series) -> str:
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"


def run_tests(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in COMPARISONS:
        subset = seed_results[seed_results["dataset"] == comp.dataset]
        left = subset[subset["model"] == comp.left_model].set_index("seed")
        right = subset[subset["model"] == comp.right_model].set_index("seed")
        shared_seeds = sorted(set(left.index).intersection(right.index))
        if len(shared_seeds) < 2:
            raise ValueError(f"Not enough shared seeds for {comp.dataset}: {shared_seeds}")

        for metric in comp.metrics:
            left_values = left.loc[shared_seeds, metric].astype(float)
            right_values = right.loc[shared_seeds, metric].astype(float)
            welch = stats.ttest_ind(left_values, right_values, equal_var=False)
            paired = stats.ttest_rel(left_values, right_values)

            higher = metric in comp.higher_is_better
            delta = left_values.mean() - right_values.mean()
            better = delta > 0 if higher else delta < 0
            rows.append({
                "dataset": comp.dataset,
                "task": comp.task,
                "metric": metric,
                "direction": "higher" if higher else "lower",
                "left_model": comp.left_model,
                "right_model": comp.right_model,
                "n": len(shared_seeds),
                "left_mean": left_values.mean(),
                "left_std": left_values.std(ddof=1),
                "right_mean": right_values.mean(),
                "right_std": right_values.std(ddof=1),
                "delta_left_minus_right": delta,
                "welch_t": welch.statistic,
                "welch_p": welch.pvalue,
                "paired_t": paired.statistic,
                "paired_p": paired.pvalue,
                "left_better": better,
                "significant_welch_0.05": bool(welch.pvalue < 0.05),
            })
    return pd.DataFrame(rows)


def p_text(p_value: float) -> str:
    if math.isnan(p_value):
        return "nan"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def write_markdown(seed_results: pd.DataFrame, test_results: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Significance Analysis From Logs",
        "",
        "KIBA is not included because it is still running. This report covers Davis, Human, and C.elegans.",
        "",
        "Primary significance test: two-sided Welch's t-test over shared seeds. A paired t-test is also exported in the CSV because each comparison uses the same seed list.",
        "",
        "## Extracted Per-Seed Results",
        "",
    ]

    for dataset in ["Davis", "Human", "C.elegans"]:
        lines.append(f"### {dataset}")
        cols = ["dataset", "model", "seed", "best_epoch"]
        metrics = ["mse", "ci", "rm2"] if dataset == "Davis" else ["AUROC", "AUPRC", "Precision", "Recall"]
        table = seed_results[seed_results["dataset"] == dataset][cols + metrics].copy()
        lines.append(table.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.extend(["## Significance Summary", ""])
    for dataset in ["Davis", "Human", "C.elegans"]:
        lines.append(f"### {dataset}")
        subset = test_results[test_results["dataset"] == dataset].copy()
        display = pd.DataFrame({
            "Metric": subset["metric"],
            "HAG-DTA": [f"{m:.4f} ± {s:.4f}" for m, s in zip(subset["left_mean"], subset["left_std"])],
            "Baseline": [f"{m:.4f} ± {s:.4f}" for m, s in zip(subset["right_mean"], subset["right_std"])],
            "Delta": [f"{x:.4f}" for x in subset["delta_left_minus_right"]],
            "Welch p": [p_text(x) for x in subset["welch_p"]],
            "Significant": ["Yes" if x else "No" for x in subset["significant_welch_0.05"]],
        })
        lines.append(display.to_markdown(index=False))
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- Davis compares HAG-DTA GIN against GraphDTA GIN using MSE, CI, and rm2. For MSE, lower is better; for CI and rm2, higher is better.",
        "- Human and C.elegans compare HAG-DTA GIN against TransformerCPI using AUROC, AUPRC, Precision, and Recall.",
        "- Accuracy, F1, and MCC are not included here because the existing logs do not contain enough prediction-level information to reconstruct them.",
        "- The extracted CSV and test summary CSV are generated next to this Markdown file.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    if not LOG_DIR.exists():
        raise FileNotFoundError(f"Missing log directory: {LOG_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_results = collect_seed_results()
    test_results = run_tests(seed_results)

    seed_path = OUTPUT_DIR / "significance_extracted_seed_results.csv"
    test_path = OUTPUT_DIR / "significance_test_results.csv"
    md_path = OUTPUT_DIR / "significance_analysis.md"

    seed_results.to_csv(seed_path, index=False)
    test_results.to_csv(test_path, index=False)
    write_markdown(seed_results, test_results, md_path)

    print(f"Wrote {seed_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
