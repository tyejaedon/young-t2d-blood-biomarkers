"""
Differential Expression Visualizations Module
Adheres to headless standards (matplotlib.use('Agg')), SEED=42, and publication-grade aesthetics.
Generates:
- Volcano plots highlighting top statistically significant candidate biomarkers
- MA plots (log2FC vs Base Mean Intensity)
- Heatmaps of top candidate biomarkers across phenotypic stages (Control, IFG, T2D)
- P-value and FDR diagnostic distributions
- Comparative low-expression filtering DEG yield evaluations
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_volcano(
    dea_df: pd.DataFrame,
    output_path: Path,
    title: str = "Volcano Plot: Discovery Cohort (GSE26168 T2D vs Control)",
    log2fc_thresh: float = 1.0,
    fdr_thresh: float = 0.05,
    top_n_annotate: int = 12
) -> Path:
    """
    Renders publication-grade volcano plot (-log10(adj_p_val) vs log2FC)
    annotating top prioritized biomarkers with non-overlapping labels.
    """
    df = dea_df.dropna(subset=["log2FC", "adj_p_val"]).copy()
    df["neg_log10_padj"] = -np.log10(np.clip(df["adj_p_val"], 1e-15, 1.0))

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)

    # Subsets
    up = df[(df["adj_p_val"] < fdr_thresh) & (df["log2FC"] >= log2fc_thresh)]
    down = df[(df["adj_p_val"] < fdr_thresh) & (df["log2FC"] <= -log2fc_thresh)]
    ns = df[~((df["adj_p_val"] < fdr_thresh) & (df["log2FC"].abs() >= log2fc_thresh))]

    # Scatter points
    ax.scatter(ns["log2FC"], ns["neg_log10_padj"], c="#b0bec5", alpha=0.45, s=20, edgecolors="none", label=f"Not Significant (n={len(ns):,})")
    ax.scatter(down["log2FC"], down["neg_log10_padj"], c="#1e88e5", alpha=0.9, s=55, edgecolors="black", linewidth=0.5, label=f"Down-regulated (n={len(down)})")
    ax.scatter(up["log2FC"], up["neg_log10_padj"], c="#e53935", alpha=0.9, s=55, edgecolors="black", linewidth=0.5, label=f"Up-regulated (n={len(up)})")

    # Cutoff threshold lines
    neg_log10_fdr_cutoff = -np.log10(fdr_thresh)
    ax.axhline(neg_log10_fdr_cutoff, color="black", linestyle="--", linewidth=1.2, alpha=0.75, label=f"FDR = {fdr_thresh}")
    ax.axvline(-log2fc_thresh, color="black", linestyle=":", linewidth=1.2, alpha=0.75, label=f"|log2FC| = {log2fc_thresh}")
    ax.axvline(log2fc_thresh, color="black", linestyle=":", linewidth=1.2, alpha=0.75)

    # Annotate top significant genes
    top_degs = df[df["is_significant"]].sort_values("adj_p_val").head(top_n_annotate)
    for _, row in top_degs.iterrows():
        gene = str(row["gene_symbol"])
        x = float(row["log2FC"])
        y = float(row["neg_log10_padj"])
        offset_x = 10 if x > 0 else -10
        ha = "left" if x > 0 else "right"
        ax.annotate(
            gene,
            xy=(x, y),
            xytext=(offset_x, 6),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            ha=ha,
            arrowprops=dict(arrowstyle="->", color="#37474f", lw=0.8, alpha=0.85)
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"$\log_2$ Fold Change (T2D vs Control)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$-\log_{10}$ (FDR Adjusted $p$-value)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cfd8dc", loc="upper left", fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_ma(
    dea_df: pd.DataFrame,
    output_path: Path,
    title: str = "MA Plot: Log2FC vs Mean Expression (GSE26168)",
    log2fc_thresh: float = 1.0,
    fdr_thresh: float = 0.05
) -> Path:
    """
    Renders MA plot (log2FC vs Base Mean intensity) to diagnose intensity-dependent bias.
    """
    df = dea_df.dropna(subset=["base_mean", "log2FC", "adj_p_val"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    sig = df[(df["adj_p_val"] < fdr_thresh) & (df["log2FC"].abs() >= log2fc_thresh)]
    ns = df[~df.index.isin(sig.index)]

    ax.scatter(ns["base_mean"], ns["log2FC"], c="#90a4ae", alpha=0.4, s=18, edgecolors="none", label=f"Not Significant (n={len(ns):,})")
    ax.scatter(sig["base_mean"], sig["log2FC"], c="#d81b60", alpha=0.9, s=45, edgecolors="black", linewidth=0.5, label=f"Significant DEG (n={len(sig)})")

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.axhline(log2fc_thresh, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(-log2fc_thresh, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)

    # Annotate top 5 DEGs
    top5 = sig.sort_values("adj_p_val").head(6)
    for _, row in top5.iterrows():
        ax.annotate(
            str(row["gene_symbol"]),
            xy=(float(row["base_mean"]), float(row["log2FC"])),
            xytext=(8, 5 if row["log2FC"] > 0 else -10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#263238", lw=0.7)
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Mean Normalized $\log_2$ Intensity", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\log_2$ Fold Change (T2D vs Control)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper right", fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_deg_heatmap(
    df_expr: pd.DataFrame,
    top_genes: List[str],
    design_df: pd.DataFrame,
    output_path: Path,
    title: str = "Clustered Expression Heatmap of Top Biomarker Candidates (GSE26168)"
) -> Path:
    """
    Renders standardized z-score heatmap of top DEGs ordered by clinical stage:
    Control (n=8) -> Impaired Fasting Glucose (n=7) -> Type 2 Diabetes (n=9).
    Demonstrates phenotypic trajectory across all 24 donor samples.
    """
    common_genes = [g for g in top_genes if g in df_expr.index]
    if not common_genes:
        return output_path

    # Align samples in logical progression order
    design_aligned = design_df.set_index("sample_id").loc[[s for s in df_expr.columns if s in design_df["sample_id"].values]]
    order_map = {"Control": 0, "IFG": 1, "T2D": 2}
    design_aligned["stage_order"] = design_aligned["condition"].map(order_map)
    sorted_samples = design_aligned.sort_values("stage_order").index.tolist()

    sub_expr = df_expr.loc[common_genes, sorted_samples]

    # Gene-wise Z-score normalization
    z_scores = sub_expr.apply(lambda row: (row - row.mean()) / (row.std(ddof=1) if row.std(ddof=1) > 1e-6 else 1.0), axis=1)

    # Condition color bars
    palette = {"Control": "#2e7d32", "IFG": "#f57c00", "T2D": "#c62828"}
    col_colors = design_aligned.loc[sorted_samples, "condition"].map(palette)

    plt.figure(figsize=(13, 8.5), dpi=300)
    g = sns.clustermap(
        z_scores,
        col_cluster=False,
        row_cluster=True,
        col_colors=col_colors,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        cbar_kws={"label": "Gene Expression Z-score"},
        figsize=(12, 8.5),
        linewidths=0.25,
        linecolor="#cfd8dc"
    )

    g.fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    g.ax_heatmap.set_xlabel("Samples (Control [Green] → IFG [Orange] → T2D [Red])", fontsize=11, fontweight="bold")
    g.ax_heatmap.set_ylabel("Prioritized Biomarker Genes", fontsize=11, fontweight="bold")

    # Add custom legend for clinical stages
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", edgecolor="black", label="Control (n=8)"),
        Patch(facecolor="#f57c00", edgecolor="black", label="Impaired Fasting Glucose (n=7)"),
        Patch(facecolor="#c62828", edgecolor="black", label="Type 2 Diabetes (n=9)")
    ]
    g.ax_heatmap.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=True, fontsize=9.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pvalue_distribution(
    dea_df: pd.DataFrame,
    output_path: Path,
    title: str = "P-value & FDR Calibration Diagnostics (GSE26168)"
) -> Path:
    """
    Renders histograms of raw p-values and moderated p-values
    to confirm expected uniform null distribution and anti-conservative signal peak at 0.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Raw p-value distribution
    p_ord = dea_df["p_value_ordinary"].dropna()
    ax1.hist(p_ord, bins=40, color="#5c6bc0", edgecolor="black", alpha=0.85)
    ax1.axhline(len(p_ord) / 40.0, color="red", linestyle="--", label="Uniform Null Expectation")
    ax1.set_title("A) Ordinary t-test P-value Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Nominal P-value", fontsize=10)
    ax1.set_ylabel("Gene Count", fontsize=10)
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Moderated p-value distribution
    p_mod = dea_df["p_value"].dropna()
    ax2.hist(p_mod, bins=40, color="#26a69a", edgecolor="black", alpha=0.85)
    ax2.axhline(len(p_mod) / 40.0, color="red", linestyle="--", label="Uniform Null Expectation")
    ax2.set_title("B) Empirical Bayes Moderated P-value Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Moderated P-value", fontsize=10)
    ax2.set_ylabel("Gene Count", fontsize=10)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_comparative_deg_yield(
    summary_df: pd.DataFrame,
    overlap_count: int,
    output_path: Path,
    title: str = "Comparative Evaluation of Low-Expression Filtering on Discovery DEA"
) -> Path:
    """
    Renders bar chart comparison of features tested, DEG yields, and overlap between
    Approach 1 (Background Floor) and Approach 2 (Distribution-Aware Variance Floor).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Features Tested vs Retained
    x = np.arange(len(summary_df))
    width = 0.35
    ax1.bar(x - width/2, summary_df["features_tested"], width, label="Total Features Tested", color="#455a64", edgecolor="black")
    bars_deg = ax1.bar(x + width/2, summary_df["deg_count"], width, label=r"Significant DEGs ($|\log_2\mathrm{FC}| \geq 1, p_{\mathrm{adj}} < 0.05$)", color="#d32f2f", edgecolor="black")

    for bar in bars_deg:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{int(yval)} DEGs", ha="center", va="bottom", fontweight="bold", color="#b71c1c")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Approach 1:\nHeuristic Background Floor", f"Approach 2:\nVariance Floor (IQR)"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Feature Count", fontsize=10, fontweight="bold")
    ax1.set_title("A) Testing Burden & Statistical DEG Yield", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: Directional Breakdown & Shared Overlap
    up_counts = summary_df["up_regulated"].values
    down_counts = summary_df["down_regulated"].values
    ax2.bar(x, up_counts, width=0.45, label="Up-regulated", color="#ef5350", edgecolor="black")
    ax2.bar(x, down_counts, width=0.45, bottom=up_counts, label="Down-regulated", color="#42a5f5", edgecolor="black")

    # Annotate shared core
    ax2.text(0.5, max(summary_df["deg_count"]) * 0.5, f"Shared Core Biomarkers:\n{overlap_count} Genes ({overlap_count / max(summary_df['deg_count']) * 100:.1f}% Concordance)",
             ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", edgecolor="#fbc02d", lw=1.5), fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Approach 1", f"Approach 2"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("DEG Count", fontsize=10, fontweight="bold")
    ax2.set_title("B) Dysregulation Directionality & Overlap", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
