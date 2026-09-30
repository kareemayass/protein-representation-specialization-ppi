"""Plot reported LoRA diagnostics from Table 1 and Figure 1.

Only aggregate values are available here. No score distributions, scatter points,
uncertainty intervals, or experimental predictions are reconstructed.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
INK = "#172b43"
BASE = "#475569"
ADAPTED = "#087f8c"
LOSS = "#b95a40"
BACKGROUND = "#f8fafc"


def style_axis(axis):
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)


def main():
    with (ROOT / "reports/lora_results.csv").open(newline="", encoding="utf-8") as stream:
        performance = list(csv.DictReader(stream))
    with (ROOT / "reports/lora_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        diagnostics = {(row["analysis"], row["quantity"]): float(row["value"]) for row in csv.DictReader(stream)}

    def value(analysis, quantity):
        return diagnostics[analysis, quantity]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "ppi-lora-diagnostics",
        "text.color": INK, "axes.labelcolor": BASE, "xtick.color": BASE,
        "axes.edgecolor": "#cbd5e1",
    })
    fig = plt.figure(figsize=(13, 11), facecolor=BACKGROUND)
    a = fig.add_axes([0.15, 0.53, 0.30, 0.22])
    b = fig.add_axes([0.66, 0.53, 0.29, 0.22])
    c = fig.add_axes([0.15, 0.20, 0.30, 0.20])
    d = fig.add_axes([0.66, 0.20, 0.29, 0.20])
    for axis in [a, b, c, d]:
        style_axis(axis)

    fig.text(0.045, 0.966, "PPI adaptation: accuracy is only part of the story", fontsize=21, weight="bold")
    fig.text(0.045, 0.934, "Pretrained ESM-2 / TUnA vs joint LoRA-r8 + TUnA · Intra2: 52,048 pairs", fontsize=12, color=BASE)

    a.set_title("A   A small accuracy gain, with trade-offs", loc="left", fontsize=12, weight="bold", pad=26)
    metric_names = ["accuracy", "auprc", "recall", "specificity"]
    for i, metric in enumerate(metric_names):
        x0, x1 = (float(row[metric]) for row in performance)
        a.plot([x0, x1], [i, i], color="#cbd5e1", linewidth=3)
        for x, offset, color in [(x0, 11, BASE), (x1, -17, ADAPTED)]:
            a.scatter(x, i, s=60, color=color, zorder=3)
            a.annotate(f"{x:.3f}", (x, i), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9, color=color)
    a.set_yticks(range(4), ["Accuracy", "AUPRC", "Recall", "Specificity"])
    a.set_ylim(3.5, -0.5)
    a.set_xlim(0.54, 0.76)
    a.set_xticks([0.55, 0.60, 0.65, 0.70, 0.75])
    a.set_xlabel("Reported metric (axis: 0.54–0.76)")
    a.legend(handles=[Line2D([], [], marker="o", linestyle="", color=BASE, label="Pretrained"),
                      Line2D([], [], marker="o", linestyle="", color=ADAPTED, label="LoRA-r8")],
             loc="lower left", bbox_to_anchor=(-0.02, 1.00), ncol=2, frameon=False, fontsize=9,
             handletextpad=0.4, columnspacing=1.2)

    corrected = int(value("pair_transitions", "false_negatives_recovered") + value("pair_transitions", "false_positives_corrected"))
    broken = int(value("pair_transitions", "true_negatives_lost") + value("pair_transitions", "true_positives_lost"))
    b.set_title("B   Corrections were offset by new errors", loc="left", fontsize=12, weight="bold", pad=26)
    b.barh([0, 1], [corrected, broken], color=[ADAPTED, LOSS], height=0.45)
    for y, count in enumerate([corrected, broken]):
        b.text(count + 70, y, f"{count:,}", va="center", fontsize=11, weight="bold")
    b.set_yticks([0, 1], ["Corrected", "New errors"])
    b.set_ylim(1.8, -0.6)
    b.set_xlim(0, 3950)
    b.set_xticks([0, 1000, 2000, 3000])
    b.set_xlabel("Number of pairs")
    b.text(0, 1.6, f"{corrected + broken:,} changes → {corrected - broken:+,} net correct", fontsize=12, weight="bold", color=INK)

    c.set_title("C   Protein identity predicts score changes", loc="left", fontsize=12, weight="bold", pad=24)
    r2 = [value("endpoint_regression", k) for k in ["pair_controls_r2", "protein_identity_r2", "combined_r2"]]
    c.barh(range(3), r2, height=0.43, color=["#94a3b8", ADAPTED, BASE])
    for i, x in enumerate(r2):
        c.text(x + 0.012, i, f"{x:.3f}", va="center", fontsize=11)
    c.set_yticks(range(3), ["Pair controls", "Protein identity", "Both"])
    c.set_ylim(2.6, -0.6)
    c.set_xlim(0, 0.55)
    c.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    c.set_xlabel("Cross-validated R² for Δscore")
    c.text(0, -0.28, "Pair-wise folds within Intra2; endpoints may recur.", transform=c.transAxes, fontsize=9, color=BASE)

    d.set_title("D   Partner ranking did not improve", loc="left", fontsize=12, weight="bold", pad=24)
    within = [value("partner_discrimination", k) for k in ["pretrained_mean_within_anchor_auroc", "lora_mean_within_anchor_auroc", "delta_score_mean_within_anchor_auroc"]]
    for i, x in enumerate(within):
        d.scatter(x, i, s=85, color=[BASE, ADAPTED, "#956caa"][i], zorder=3)
        d.annotate(f"{x:.3f}", (x, i), xytext=(9, 0), textcoords="offset points", va="center")
    d.set_yticks(range(3), ["Pretrained", "LoRA-r8", "Δscore alone"])
    d.set_ylim(2.6, -0.6)
    d.set_xlim(0.50, 0.76)
    d.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    d.set_xlabel("Mean within-anchor AUROC (axis: 0.50–0.76)")
    d.text(0, -0.28, "Positive vs negative partners of the same protein.", transform=d.transAxes, fontsize=9, color=BASE)

    n = int(value("unseen_protein_transfer", "n_proteins"))
    transfer_r2 = value("unseen_protein_transfer", "r2")
    rho = value("unseen_protein_transfer", "spearman_rho")
    fig.text(0.045, 0.884, "Endpoint effects transfer across protein-disjoint splits", fontsize=15, weight="bold")
    fig.text(0.045, 0.858, f"Frozen ESM-2 → PCA–ridge fitted on Intra0 → {n:,} unseen Intra2 proteins", fontsize=11, color=BASE)
    fig.text(0.045, 0.830, f"R² = {transfer_r2:.3f}  ·  Spearman ρ = {rho:.3f}", fontsize=14, color=ADAPTED, weight="bold")
    fig.text(0.045, 0.080, "Generalization to unseen proteins did not produce better discrimination among their partners.", fontsize=12, weight="bold")
    fig.text(0.045, 0.054, "Transfer above and within-Intra2 regression (C) are separate analyses. Joint adaptation effects are not attributable to LoRA alone.", fontsize=9, color=BASE)
    fig.text(0.045, 0.028, "Source: project report, Table 1 and Figure 1. Reported aggregates; no distributions or uncertainty intervals reconstructed.", fontsize=9, color=BASE)
    output = ROOT / "docs/assets/lora-diagnostics.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None}, facecolor=BACKGROUND)
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
