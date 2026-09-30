"""Render the README figure from transcribed report metrics, not raw predictions."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with (ROOT / "reports/transfer_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "ppi-transfer-summary",
        "axes.edgecolor": "#cbd5e1", "text.color": "#172b43",
        "axes.labelcolor": "#475569", "xtick.color": "#475569",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#334155", "#b95a40", "#956caa", "#087f8c"]
    labels = ["Base ESMC", "Domain specialization", "+ MLM preservation", "+ Representation preservation"]
    for axis, metric, title, subtitle in zip(
        axes,
        ["ppi_auprc", "domain_end_to_end_f1"],
        ["PPI transfer", "Domain prediction"],
        ["Full Intra2 · 52,048 pairs", "Held-out InterPro · end-to-end F1"],
    ):
        axis.set_facecolor("#f8fafc")
        for i, row in enumerate(rows):
            if row[metric]:
                value = float(row[metric])
                axis.scatter(value, i, s=100, color=colors[i], zorder=3)
                axis.annotate(f"{value:.4f}", (value, i), xytext=(9, 0),
                              textcoords="offset points", va="center", fontsize=11)
            else:
                axis.text(0.57, i, "—", va="center", color="#94a3b8")
        axis.set_ylim(3.55, -0.6)
        axis.set_xlim(0.55, 0.85)
        axis.set_xticks([0.55, 0.65, 0.75, 0.85])
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_yticks(range(4), labels if axis is axes[0] else [""] * 4)
        axis.tick_params(axis="y", length=0, pad=12)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_title(title, loc="left", fontsize=15, weight="bold", pad=30)
        axis.text(0, 1.045, subtitle, transform=axis.transAxes, fontsize=10, color="#64748b")
        axis.set_xlabel("AUPRC · higher is better" if metric == "ppi_auprc" else "F1 · higher is better", labelpad=10)
    fig.subplots_adjust(left=0.30, right=0.975, top=0.77, bottom=0.22, wspace=0.30)
    fig.text(0.035, 0.945, "Biological specialization and downstream transfer", fontsize=19, weight="bold")
    fig.text(0.035, 0.075, "Reported aggregate results, not rerun experiments. Different tasks and test populations. Axes show 0.55–0.85.", fontsize=9, color="#64748b")
    fig.text(0.035, 0.035, "— = not transcribed / not reported here. Source: project report, Results and Figure 2; reports/transfer_results.csv.", fontsize=9, color="#64748b")
    output = ROOT / "docs/assets/transfer-summary.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None}, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
