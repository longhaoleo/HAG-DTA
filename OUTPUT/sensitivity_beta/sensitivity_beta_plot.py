import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = "./mmd_beta_human_summary.csv"
df = pd.read_csv(csv_path)

df["beta_label"] = df["beta"].astype(str)
x = range(len(df))

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

colors = {
    "AUROC": "#1F4E79",
    "AUPRC": "#2A9D8F",
    "Precision": "#3A7D44",
    "Recall": "#64B5CD"
}

markers = {
    "AUROC": "o",
    "AUPRC": "s",
    "Precision": "^",
    "Recall": "D"
}

def plot_beta_sensitivity(metric_list, filename, title):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for metric in metric_list:
        ax.plot(
            x,
            df[metric],
            label=metric,
            color=colors[metric],
            marker=markers[metric],
            linewidth=2.2,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.6
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["beta_label"])

    ax.set_xlabel(r"$\beta$", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)

    y_min = df[metric_list].min().min() - 0.004
    y_max = df[metric_list].max().max() + 0.004
    ax.set_ylim(y_min, y_max)

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=1
    )

    legend = ax.legend(
        frameon=True,
        fontsize=10,
        loc="best"
    )
    legend.get_frame().set_edgecolor("#CCCCCC")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()

    out_dir = Path("./figures")
    out_dir.mkdir(exist_ok=True)

    plt.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{filename}.png", dpi=600, bbox_inches="tight")

    plt.show()


plot_beta_sensitivity(
    ["AUROC", "AUPRC"],
    "beta_sensitivity_auc",
    r"Sensitivity Analysis of $\beta$ on AUROC and AUPRC"
)

plot_beta_sensitivity(
    ["Precision", "Recall"],
    "beta_sensitivity_pr",
    r"Sensitivity Analysis of $\beta$ on Precision and Recall"
)