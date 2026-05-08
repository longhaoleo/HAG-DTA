#!/usr/bin/env python3
"""
process_experiment_results.py

从 HAG-DTA 的训练日志与 random.csv 中提取实验结果，生成可复查的 Markdown 汇总。
默认读取：
  - OUTPUT/logs/*.log
  - OUTPUT/*_random.csv
默认输出：
  - experiment_results_summary.md

说明：
1. 回归任务后续主口径统一为 n1=4、n2=2。Davis 的主结果采用 LOG/实验情况报告.md 中整理后的回归结果；默认不把历史 OUTPUT Davis 日志纳入主表。
   若需要内部排查旧 Davis 日志，可执行：python process_experiment_results.py --include-current-davis
2. Human / Celegans 分类任务将重新运行；默认不把历史分类日志纳入主表。
   若需要内部排查旧分类日志，可执行：python process_experiment_results.py --include-classification
3. KIBA 当前耗时较长，论文主表默认留空；如果后续要解析 KIBA，可执行：
   python process_experiment_results.py --include-kiba
4. 脚本不会修改论文 md，只生成独立汇总，便于人工核对后再写入正文。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

SEEDS = [100, 1000, 2000, 3000, 4000]

DATASET_ORDER = ["davis", "kiba", "Human", "Celegans"]
REGRESSION_DATASETS = {"davis", "kiba"}
CLASSIFICATION_DATASETS = {"Human", "Celegans"}

REGRESSION_FINAL_RE = re.compile(
    r"final test metrics at best val\s+(?P<criterion>MSE|CI)\s+epoch\s+"
    r"(?P<epoch>\d+)\s*:\s*mse=\s*(?P<mse>[0-9.eE+-]+)\s+"
    r"ci=\s*(?P<ci>[0-9.eE+-]+).*?rm2=\s*(?P<r2>[0-9.eE+-]+)",
    flags=re.DOTALL,
)

CLASSIFICATION_BEST_RE = re.compile(
    r"best_AUROC, best_AUPRC, best_precision, best_recall:\s*"
    r"(?P<AUROC>[0-9.eE+-]+)\s+"
    r"(?P<AUPRC>[0-9.eE+-]+)\s+"
    r"(?P<Precision>[0-9.eE+-]+)\s+"
    r"(?P<Recall>[0-9.eE+-]+)"
)


@dataclass
class ResultBundle:
    dataset: str
    rows: pd.DataFrame
    source: str


def read_text_without_nul(path: Path) -> str:
    """读取日志并去掉 NUL 字节，避免 tail/cat 被污染日志影响。"""
    data = path.read_bytes()
    return data.replace(b"\x00", b"").decode("utf-8", errors="replace")


def parse_fold_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_f(\d+)\.log$|fold(\d+)", path.name)
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def parse_regression_logs(log_dir: Path, dataset: str) -> ResultBundle:
    rows: List[dict] = []
    for log_file in sorted(log_dir.glob(f"{dataset}_f*.log")):
        fold = parse_fold_from_name(log_file)
        if fold is None:
            continue
        text = read_text_without_nul(log_file)
        matches = list(REGRESSION_FINAL_RE.finditer(text))
        for idx, m in enumerate(matches):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": SEEDS[idx] if idx < len(SEEDS) else None,
                    "criterion": m.group("criterion"),
                    "best_epoch": int(m.group("epoch")),
                    "mse": float(m.group("mse")),
                    "ci": float(m.group("ci")),
                    "r2": float(m.group("r2")),
                    "log_file": log_file.name,
                }
            )
    return ResultBundle(dataset=dataset, rows=pd.DataFrame(rows), source="logs/final test metrics")


def parse_classification_logs(log_dir: Path, dataset: str) -> ResultBundle:
    rows: List[dict] = []
    for log_file in sorted(log_dir.glob(f"{dataset.lower()}_f*.log")):
        fold = parse_fold_from_name(log_file)
        if fold is None:
            continue
        text = read_text_without_nul(log_file)
        # 每个 seed 都会重新打印 running on <dataset>，按该标记分段。
        segments = [seg for seg in re.split(rf"(?=running on\s+{re.escape(dataset)})", text) if f"running on  {dataset}" in seg or f"running on {dataset}" in seg]
        for idx, seg in enumerate(segments):
            matches = list(CLASSIFICATION_BEST_RE.finditer(seg))
            if not matches:
                continue
            m = matches[-1]
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": SEEDS[idx] if idx < len(SEEDS) else None,
                    "AUROC": float(m.group("AUROC")),
                    "AUPRC": float(m.group("AUPRC")),
                    "Precision": float(m.group("Precision")),
                    "Recall": float(m.group("Recall")),
                    "log_file": log_file.name,
                }
            )
    return ResultBundle(dataset=dataset, rows=pd.DataFrame(rows), source="logs/best_AUROC per seed")


def parse_random_csv(output_dir: Path, dataset: str) -> ResultBundle:
    rows: List[dict] = []
    for csv_file in sorted(output_dir.glob(f"{dataset}_*_random.csv")):
        fold_match = re.search(r"fold(\d+)_random", csv_file.name)
        if not fold_match:
            continue
        fold = int(fold_match.group(1))
        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            d = row.to_dict()
            d.update(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": SEEDS[idx] if idx < len(SEEDS) else None,
                    "csv_file": csv_file.name,
                }
            )
            rows.append(d)
    return ResultBundle(dataset=dataset, rows=pd.DataFrame(rows), source="random.csv")


def mean_std(df: pd.DataFrame, metrics: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    return {m: (float(df[m].mean()), float(df[m].std(ddof=1))) for m in metrics if m in df.columns}


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def fmt_pm(mu: float, sd: float, digits: int = 4) -> str:
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def dataframe_markdown(df: pd.DataFrame) -> str:
    """不用 pandas.to_markdown，避免额外依赖 tabulate。"""
    if df.empty:
        return ""
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]):
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else fmt(float(x), 6))
        else:
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else str(x))
    return markdown_table(list(safe.columns), safe.values.tolist())


def bundle_for_dataset(output_dir: Path, log_dir: Path, dataset: str, include_kiba: bool, include_current_davis: bool, include_current_classification: bool) -> ResultBundle:
    if dataset in REGRESSION_DATASETS:
        if dataset == "kiba" and not include_kiba:
            return ResultBundle(dataset=dataset, rows=pd.DataFrame(), source="left blank")
        bundle = parse_regression_logs(log_dir, dataset)
        if not bundle.rows.empty:
            return bundle
        return parse_random_csv(output_dir, dataset)

    if dataset in CLASSIFICATION_DATASETS:
        if not include_current_classification:
            return ResultBundle(dataset=dataset, rows=pd.DataFrame(), source="pending rerun with n1=4,n2=2")
        bundle = parse_classification_logs(log_dir, dataset)
        if not bundle.rows.empty:
            return bundle
        return parse_random_csv(output_dir, dataset)

    return parse_random_csv(output_dir, dataset)


def parse_sensitivity_davis(sensitivity_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for log_file in sorted(sensitivity_dir.glob("n1_*_n2_*.log")):
        m_name = re.match(r"n1_(\d+)_n2_(\d+)\.log", log_file.name)
        if not m_name:
            continue
        text = read_text_without_nul(log_file)
        matches = list(REGRESSION_FINAL_RE.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        rows.append(
            {
                "n1": int(m_name.group(1)),
                "n2": int(m_name.group(2)),
                "best_epoch": int(m.group("epoch")),
                "mse": float(m.group("mse")),
                "ci": float(m.group("ci")),
                "r2": float(m.group("r2")),
                "log_file": log_file.name,
            }
        )
    return pd.DataFrame(rows).sort_values(["n1", "n2"]) if rows else pd.DataFrame()


def parse_sensitivity_human(sensitivity_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for log_file in sorted(sensitivity_dir.glob("n1_*_n2_*.log")):
        m_name = re.match(r"n1_(\d+)_n2_(\d+)\.log", log_file.name)
        if not m_name:
            continue
        text = read_text_without_nul(log_file)
        matches = list(CLASSIFICATION_BEST_RE.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        rows.append(
            {
                "n1": int(m_name.group(1)),
                "n2": int(m_name.group(2)),
                "AUROC": float(m.group("AUROC")),
                "AUPRC": float(m.group("AUPRC")),
                "Precision": float(m.group("Precision")),
                "Recall": float(m.group("Recall")),
                "log_file": log_file.name,
            }
        )
    return pd.DataFrame(rows).sort_values(["n1", "n2"]) if rows else pd.DataFrame()


def parse_mmd_sensitivity(sensitivity_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for log_file in sorted(sensitivity_dir.glob("mmd_beta_*.log")):
        beta = log_file.stem.replace("mmd_beta_", "")
        text = read_text_without_nul(log_file)
        matches = list(REGRESSION_FINAL_RE.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        rows.append(
            {
                "beta": beta,
                "best_epoch": int(m.group("epoch")),
                "mse": float(m.group("mse")),
                "ci": float(m.group("ci")),
                "r2": float(m.group("r2")),
                "log_file": log_file.name,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["beta_sort"] = df["beta"].astype(float)
    return df.sort_values("beta_sort").drop(columns=["beta_sort"])


def build_report(root: Path, include_kiba: bool, include_current_davis: bool, include_current_classification: bool) -> str:
    output_dir = root / "OUTPUT"
    log_dir = output_dir / "logs"

    bundles = [bundle_for_dataset(output_dir, log_dir, ds, include_kiba, include_current_davis, include_current_classification) for ds in DATASET_ORDER]

    lines: List[str] = []
    lines.append("# HAG-DTA 实验结果自动汇总")
    lines.append("")
    lines.append(f"数据根目录：`{root}`")
    lines.append(f"日志目录：`{log_dir}`")
    lines.append(f"结果目录：`{output_dir}`")
    lines.append("")
    lines.append("说明：Davis 回归结果直接汇总；KIBA 在 `--include-kiba` 未开启时留空；分类在 `--include-current-classification` 未开启时待重跑。")
    lines.append("")

    # 主结果表
    regression_rows: List[List[str]] = []
    classification_rows: List[List[str]] = []
    for bundle in bundles:
        df = bundle.rows
        ds = bundle.dataset
        if ds in REGRESSION_DATASETS:
            if df.empty:
                regression_rows.append([ds, "", "", "", "", bundle.source])
            else:
                stats = mean_std(df, ["mse", "ci", "r2"])
                regression_rows.append(
                    [
                        ds,
                        str(len(df)),
                        fmt_pm(*stats["mse"]),
                        fmt_pm(*stats["ci"]),
                        fmt_pm(*stats["r2"]),
                        bundle.source,
                    ]
                )
        else:
            if df.empty:
                classification_rows.append([ds, "", "", "", "", "", bundle.source])
            else:
                stats = mean_std(df, ["AUROC", "AUPRC", "Precision", "Recall"])
                classification_rows.append(
                    [
                        ds,
                        str(len(df)),
                        fmt_pm(*stats["AUROC"]),
                        fmt_pm(*stats["AUPRC"]),
                        fmt_pm(*stats["Precision"]),
                        fmt_pm(*stats["Recall"]),
                        bundle.source,
                    ]
                )

    lines.append("## 1. 回归主结果")
    lines.append(markdown_table(["Dataset", "N", "MSE", "CI", "rm²", "Source"], regression_rows))
    lines.append("")
    lines.append("## 2. 分类主结果")
    lines.append(markdown_table(["Dataset", "N", "AUROC", "AUPRC", "Precision", "Recall", "Source"], classification_rows))
    lines.append("")

    # 每个 fold 均值，用于检查是否缺失。
    lines.append("## 3. Per-fold 检查表")
    for bundle in bundles:
        df = bundle.rows
        if df.empty:
            lines.append(f"\n### {bundle.dataset}\n无可汇总结果。")
            continue
        if bundle.dataset in REGRESSION_DATASETS:
            metrics = ["mse", "ci", "r2"]
        else:
            metrics = ["AUROC", "AUPRC", "Precision", "Recall"]
        rows: List[List[str]] = []
        for fold, g in df.groupby("fold"):
            stats = mean_std(g, metrics)
            rows.append([str(fold), str(len(g))] + [fmt_pm(*stats[m]) for m in metrics])
        lines.append(f"\n### {bundle.dataset}")
        lines.append(markdown_table(["fold", "N"] + metrics, rows))
    lines.append("")

    # 敏感性分析
    davis_sens = parse_sensitivity_davis(output_dir / "sensitivity")
    human_sens = parse_sensitivity_human(output_dir / "sensitivity_human")
    mmd_sens = parse_mmd_sensitivity(output_dir / "sensitivity_mmd")

    lines.append("## 4. n1/n2 敏感性分析")
    if not davis_sens.empty:
        rows = [
            [str(int(r.n1)), str(int(r.n2)), fmt(r.mse), fmt(r.ci), fmt(r.r2), str(int(r.best_epoch))]
            for r in davis_sens.itertuples(index=False)
        ]
        lines.append("\n### Davis")
        lines.append(markdown_table(["n1", "n2", "MSE", "CI", "rm²", "best epoch"], rows))
        best_mse = davis_sens.sort_values("mse").iloc[0]
        best_ci = davis_sens.sort_values("ci", ascending=False).iloc[0]
        lines.append(f"\nDavis 最低 MSE：n1={int(best_mse.n1)}, n2={int(best_mse.n2)}, MSE={fmt(best_mse.mse)}, CI={fmt(best_mse.ci)}。")
        lines.append(f"Davis 最高 CI：n1={int(best_ci.n1)}, n2={int(best_ci.n2)}, MSE={fmt(best_ci.mse)}, CI={fmt(best_ci.ci)}。")
    if not human_sens.empty:
        rows = [
            [str(int(r.n1)), str(int(r.n2)), fmt(r.AUROC), fmt(r.AUPRC), fmt(r.Precision), fmt(r.Recall)]
            for r in human_sens.itertuples(index=False)
        ]
        lines.append("\n### Human")
        lines.append(markdown_table(["n1", "n2", "AUROC", "AUPRC", "Precision", "Recall"], rows))
        best_auc = human_sens.sort_values("AUROC", ascending=False).iloc[0]
        best_pr = human_sens.sort_values("Precision", ascending=False).iloc[0]
        lines.append(f"\nHuman 最高 AUROC：n1={int(best_auc.n1)}, n2={int(best_auc.n2)}, AUROC={fmt(best_auc.AUROC)}。")
        lines.append(f"Human 最高 Precision：n1={int(best_pr.n1)}, n2={int(best_pr.n2)}, Precision={fmt(best_pr.Precision)}。")
    lines.append("")

    lines.append("## 5. MMD β 敏感性分析")
    if not mmd_sens.empty:
        rows = [
            [str(r.beta), fmt(r.mse), fmt(r.ci), fmt(r.r2), str(int(r.best_epoch))]
            for r in mmd_sens.itertuples(index=False)
        ]
        lines.append(markdown_table(["β", "MSE", "CI", "rm²", "best epoch"], rows))
        best_mse = mmd_sens.sort_values("mse").iloc[0]
        best_ci = mmd_sens.sort_values("ci", ascending=False).iloc[0]
        lines.append(f"\n最低 MSE：β={best_mse.beta}, MSE={fmt(best_mse.mse)}, CI={fmt(best_mse.ci)}。")
        lines.append(f"最高 CI：β={best_ci.beta}, MSE={fmt(best_ci.mse)}, CI={fmt(best_ci.ci)}。")
    else:
        lines.append("未发现 MMD 敏感性日志。")
    lines.append("")

    lines.append("## 6. 原始记录明细")
    for bundle in bundles:
        if bundle.rows.empty:
            continue
        show_cols = [c for c in ["dataset", "fold", "seed", "best_epoch", "criterion", "mse", "ci", "r2", "AUROC", "AUPRC", "Precision", "Recall", "log_file"] if c in bundle.rows.columns]
        lines.append(f"\n### {bundle.dataset}")
        lines.append(dataframe_markdown(bundle.rows[show_cols]))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HAG-DTA experiment logs and generate Markdown summary.")
    parser.add_argument("--root", default=".", help="HAG-DTA project root. Default: current directory.")
    parser.add_argument("--include-kiba", action="store_true", help="Parse KIBA results instead of leaving KIBA blank.")
    parser.add_argument("--include-current-classification", action="store_true", help="Include existing classification logs; default leaves Human/C.elegans pending rerun.")
    parser.add_argument("--include-current-davis", action="store_true", help="Include existing Davis logs for internal diagnostics; default leaves Davis pending rerun with n1=4,n2=2.")
    parser.add_argument("--output", default="experiment_results_summary.md", help="Output Markdown path relative to root unless absolute.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    report = build_report(root=root, include_kiba=args.include_kiba, include_current_davis=args.include_current_davis, include_current_classification=args.include_current_classification)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
