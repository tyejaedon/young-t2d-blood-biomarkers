"""
Biomarker Modeling Visualizations Module
Adheres to headless standards (matplotlib.use('Agg')), SEED=42, and publication-grade aesthetics.
Generates:
- Discovery cross-validation ROC curves (LOOCV)
- Independent validation ROC curves on GSE15932
- Biomarker candidate shortlist feature ranking and selection stability
- Panel scaling dynamics (Overfitting boundary & parsimony inflection)
- Pre-diabetic IFG risk trajectory score distributions
- Confusion matrices for discovery and external cohorts
- Cross-cohort biomarker expression profile comparisons
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_discovery_cv_roc(
    sample_details: Dict[str, Dict[str, Any]],
    output_path: Path,
    panel_name: str = "Core-5 Biomarker Panel"
) -> Path:
    """
    Renders ROC curves for all models evaluated under Discovery Leave-One-Out Cross-Validation.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    colors = {
        "Logistic Regression": "#1e88e5",
        "Random Forest": "#43a047",
        "SVM (Linear)": "#fb8c00",
        "SVM (RBF)": "#8e24aa"
    }

    for name, data in sample_details.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = colors.get(name, "#333333")
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#9e9e9e", linestyle="--", lw=1.2, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title(f"Discovery Cross-Validation ROC Curves (GSE26168 LOOCV)\n{panel_name}", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_independent_validation_roc(
    val_predictions: Dict[str, Dict[str, Any]],
    output_path: Path,
    panel_name: str = "Core-5 Biomarker Panel"
) -> Path:
    """
    Renders ROC curves for all models evaluated on independent GSE15932 cohort.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    colors = {
        "Logistic Regression": "#1e88e5",
        "Random Forest": "#43a047",
        "SVM (Linear)": "#fb8c00",
        "SVM (RBF)": "#8e24aa"
    }

    for name, data in val_predictions.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = colors.get(name, "#333333")
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#9e9e9e", linestyle="--", lw=1.2, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title(f"Independent Validation ROC Curves (GSE15932 PBMC Array, n=32)\n{panel_name}", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_biomarker_shortlist_importance(
    shortlist_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15
) -> Path:
    """
    Renders horizontal bar charts showing selection frequency and feature importance
    with regulation directionality coloring.
    """
    df = shortlist_df.head(top_n).copy()
    df = df.sort_values("priority_rank", ascending=False)  # Bottom-to-top in horizontal bar

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=300, sharey=True)

    # Direction colors
    colors = ["#e53935" if r == "Up-regulated" else "#1e88e5" for r in df["regulation"]]

    # Stability selection frequency
    ax1.barh(df["gene_symbol"], df["selection_frequency"] * 100, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("Cross-Validation Selection Stability (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Resampling Selection Stability", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax1.set_xlim([0, 105])

    # Random forest importance
    ax2.barh(df["gene_symbol"], df["rf_importance"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax2.set_xlabel("Random Forest Gini Importance", fontsize=10, fontweight="bold")
    ax2.set_title("B. Random Forest Gini Impurity Importance", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Custom legend for regulation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e53935", edgecolor="black", label="Up-regulated in T2D"),
        Patch(facecolor="#1e88e5", edgecolor="black", label="Down-regulated in T2D")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True, fontsize=10)

    plt.suptitle("Prioritized Candidate Biomarker Shortlist & Feature Selection Metrics", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_panel_scaling_auc(
    scaling_df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Renders line plot comparing Discovery CV AUC and External Validation AUC
    as panel size expands from 1 to 18 genes, highlighting the parsimony peak.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)

    # Average across models for clean trends
    agg = scaling_df.groupby("panel_size").agg({
        "discovery_cv_auc": "mean",
        "validation_auc": "mean"
    }).reset_index()

    ax.plot(agg["panel_size"], agg["discovery_cv_auc"], marker="o", color="#1e88e5", lw=2.4, label="Discovery CV AUC (GSE26168 LOOCV Mean)")
    ax.plot(agg["panel_size"], agg["validation_auc"], marker="s", color="#e53935", lw=2.4, label="Independent Validation AUC (GSE15932 Mean)")

    # Highlight optimal panel zones
    ax.axvspan(4.5, 8.5, color="#c8e6c9", alpha=0.45, label="Optimal Parsimony Window (5–8 Genes)")
    ax.axvline(5, color="#2e7d32", linestyle="--", lw=1.2, alpha=0.8)
    ax.axvline(8, color="#2e7d32", linestyle="--", lw=1.2, alpha=0.8)

    ax.set_xticks(range(1, 19))
    ax.set_xlabel("Number of Features in Biomarker Panel (k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Area Under ROC Curve (AUC)", fontsize=11, fontweight="bold")
    ax.set_ylim([0.45, 1.05])
    ax.set_title("Biomarker Panel Scaling Dynamics: Parsimony Peak vs Overfitting Degradation", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ifg_risk_trajectory(
    trajectory_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Renders boxplot and swarm plot of predicted T2D risk probability across
    Control -> IFG (Pre-Diabetes) -> Type 2 Diabetes stages.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    # Focus on Logistic Regression & Random Forest
    sub = trajectory_df[trajectory_df["model"].isin(["Logistic Regression", "Random Forest"])].copy()

    palette = {"Control": "#4caf50", "IFG (Pre-Diabetes)": "#ff9800", "Type 2 Diabetes": "#f44336"}

    sns.boxplot(
        data=sub,
        x="stage",
        y="predicted_t2d_risk",
        hue="model",
        palette=["#90caf9", "#a5d6a7"],
        ax=ax,
        width=0.6,
        boxprops=dict(alpha=0.8)
    )

    sns.stripplot(
        data=sub,
        x="stage",
        y="predicted_t2d_risk",
        hue="model",
        dodge=True,
        jitter=0.2,
        size=6,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    # Format annotations
    rho_lr = corr_df[corr_df["model"] == "Logistic Regression"]["spearman_rho"].iloc[0]
    p_lr = corr_df[corr_df["model"] == "Logistic Regression"]["p_value"].iloc[0]
    rho_rf = corr_df[corr_df["model"] == "Random Forest"]["spearman_rho"].iloc[0]
    p_rf = corr_df[corr_df["model"] == "Random Forest"]["p_value"].iloc[0]

    ax.text(
        0.03, 0.92,
        f"LR Trend: Spearman $\\rho = {rho_lr:.2f}$ ($p = {p_lr:.1e}$)\nRF Trend: Spearman $\\rho = {rho_rf:.2f}$ ($p = {p_rf:.1e}$)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9)
    )

    ax.set_xlabel("Clinical Glycemic Progression Stage", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Type 2 Diabetes Risk Probability", fontsize=11, fontweight="bold")
    ax.set_title("Early Diagnostic Trajectory: Predicted T2D Risk Score across Glycemic Stages\nNormoglycemic Controls (n=8) $\\rightarrow$ IFG Pre-Diabetes (n=7) $\\rightarrow$ T2D (n=9)", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([-0.05, 1.08])

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    disc_details: Dict[str, Dict[str, Any]],
    val_predictions: Dict[str, Dict[str, Any]],
    output_path: Path,
    model_name: str = "Logistic Regression"
) -> Path:
    """
    Renders side-by-side confusion matrices for Discovery LOOCV and External Validation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # 1. Discovery LOOCV
    d_data = disc_details[model_name]
    cm_disc = confusion_matrix(d_data["y_true"], d_data["y_pred"])
    sns.heatmap(cm_disc, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1, annot_kws={"size": 14, "weight": "bold"})
    ax1.set_xticklabels(["Control", "T2D"])
    ax1.set_yticklabels(["Control", "T2D"])
    ax1.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Clinical Label", fontsize=11, fontweight="bold")
    ax1.set_title(f"A. Discovery LOOCV (GSE26168, n=17)\n{model_name}", fontsize=11, fontweight="bold")

    # 2. Independent Validation
    v_data = val_predictions[model_name]
    cm_val = confusion_matrix(v_data["y_true"], v_data["y_pred"])
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax2, annot_kws={"size": 14, "weight": "bold"})
    ax2.set_xticklabels(["Control", "T2D"])
    ax2.set_yticklabels(["Control", "T2D"])
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Clinical Label", fontsize=11, fontweight="bold")
    ax2.set_title(f"B. Independent Validation (GSE15932, n=32)\n{model_name}", fontsize=11, fontweight="bold")

    plt.suptitle("Supervised Classification Confusion Matrices across Discovery and Validation", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_biomarker_cross_cohort_profiles(
    disc_df: pd.DataFrame,
    val_df: pd.DataFrame,
    core_genes: List[str],
    output_path: Path
) -> Path:
    """
    Renders standardized expression comparisons for top core candidate biomarkers
    between Control and T2D in both discovery (GSE26168) and validation (GSE15932).
    """
    n_genes = len(core_genes)
    cols = 3
    rows = int(np.ceil(n_genes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6), dpi=300, sharey=False)
    axes = np.array(axes).reshape(-1)

    # Standardize within cohort for clean visual comparison
    d_std = disc_df.copy()
    v_std = val_df.copy()

    for g in core_genes:
        d_std[g] = (d_std[g] - d_std[g].mean()) / d_std[g].std()
        v_std[g] = (v_std[g] - v_std[g].mean()) / v_std[g].std()

    for idx, g in enumerate(core_genes):
        ax = axes[idx]
        plot_data = []

        for _, r in d_std.iterrows():
            grp = "T2D" if r["y"] == 1 else "Control"
            plot_data.append({"cohort": "GSE26168 (Disc)", "condition": grp, "expression_z": r[g]})

        for _, r in v_std.iterrows():
            grp = "T2D" if r["y"] == 1 else "Control"
            plot_data.append({"cohort": "GSE15932 (Val)", "condition": grp, "expression_z": r[g]})

        df_p = pd.DataFrame(plot_data)
        sns.boxplot(
            data=df_p,
            x="cohort",
            y="expression_z",
            hue="condition",
            palette={"Control": "#42a5f5", "T2D": "#ef5350"},
            ax=ax,
            width=0.5,
            boxprops=dict(alpha=0.75)
        )
        sns.stripplot(
            data=df_p,
            x="cohort",
            y="expression_z",
            hue="condition",
            dodge=True,
            size=5,
            edgecolor="black",
            linewidth=0.5,
            ax=ax
        )

        ax.set_title(f"{g}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Standardized Expression (z-score)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Remove inner duplicate legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    # Single global legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#42a5f5", edgecolor="black", label="Control"),
        Patch(facecolor="#ef5350", edgecolor="black", label="Type 2 Diabetes")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True, fontsize=10)

    plt.suptitle("Cross-Cohort Standardized Expression Profiles of Core Candidate Biomarkers", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
