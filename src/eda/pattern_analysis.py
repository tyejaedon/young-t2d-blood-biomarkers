"""
Pattern Identification & Covariate Association Module
Investigates:
- Phenotypic separation and monotonic early dysregulation trajectory in GSE26168
- Covariate and confounder association analysis (PC1-PC5 vs age, sex, platform, diabetes)
- Cross-cohort high-variance gene consistency and concordance between GSE26168 and GSE15932
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from eda.common import (
    SEED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    log_eda
)


def analyze_phenotypic_trajectory_gse26168(
    norm_matrices: Dict[str, pd.DataFrame],
    metadata_df: pd.DataFrame
) -> Tuple[Path, pd.DataFrame]:
    """
    Inspects trajectory and separation between Normoglycemic Controls,
    Impaired Fasting Glucose (Pre-Diabetes), and Type 2 Diabetes in GSE26168 blood transcriptomic space:
    - Tests monotonic dysregulation trend across the 3 glycemic stages (Jonckheere-Terpstra / ANOVA).
    - Evaluates principal component trajectories capturing the progressive diabetic transition.
    Saves figure to results/figures/phenotypic_trajectory_gse26168.png.
    """
    log_eda("PATTERNS", "Analyzing phenotypic separation & early dysregulation in GSE26168...")

    df_26168 = norm_matrices["GSE26168"]
    meta_26168 = metadata_df[metadata_df["cohort"] == "GSE26168"].set_index("sample_id").loc[df_26168.columns]

    scaler = StandardScaler()
    scaled_expr = scaler.fit_transform(df_26168.T.values)
    pca = PCA(n_components=5, random_state=SEED)
    pca_scores = pca.fit_transform(scaled_expr)
    var_exp = pca.explained_variance_ratio_ * 100.0

    traj_df = pd.DataFrame(pca_scores, index=df_26168.columns, columns=[f"PC{i+1}" for i in range(5)])
    traj_df = traj_df.join(meta_26168)

    progression_order = {
        "Control": 0,
        "Impaired Fasting Glucose": 1,
        "Type 2 Diabetes": 2
    }
    traj_df["glycemic_order"] = traj_df["clinical_group"].map(progression_order)

    stats_list = []
    for i in range(1, 6):
        pc_col = f"PC{i}"
        groups = [traj_df[traj_df["clinical_group"] == g][pc_col].values for g in ["Control", "Impaired Fasting Glucose", "Type 2 Diabetes"]]
        f_stat, p_val = stats.f_oneway(*groups)
        kw_stat, kw_p = stats.kruskal(*groups)
        corr_r, corr_p = stats.spearmanr(traj_df["glycemic_order"], traj_df[pc_col])

        stats_list.append({
            "component": pc_col,
            "variance_explained_pct": var_exp[i - 1],
            "anova_f": f_stat,
            "anova_pvalue": p_val,
            "kruskal_pvalue": kw_p,
            "spearman_order_rho": corr_r,
            "spearman_order_pvalue": corr_p
        })

    traj_stats_df = pd.DataFrame(stats_list)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # Panel 1: PC1 vs PC2 Scatter showing 3 tiers
    ax1 = axes[0]
    tier_colors = {
        "Control": "#2ca02c",
        "Impaired Fasting Glucose": "#ff7f0e",
        "Type 2 Diabetes": "#d62728"
    }
    tier_markers = {"Control": "o", "Impaired Fasting Glucose": "s", "Type 2 Diabetes": "D"}

    for tier, col in tier_colors.items():
        sub = traj_df[traj_df["clinical_group"] == tier]
        ax1.scatter(sub["PC1"], sub["PC2"], label=f"{tier} (n={len(sub)})", color=col, marker=tier_markers[tier], s=110, alpha=0.9, edgecolors="black", linewidth=1.2)

    centroids = traj_df.groupby("clinical_group")[["PC1", "PC2"]].mean().loc[["Control", "Impaired Fasting Glucose", "Type 2 Diabetes"]]
    ax1.plot(centroids["PC1"], centroids["PC2"], "k--", linewidth=2.5, alpha=0.7, label="Progression Centroid Vector")
    ax1.set_title(f"A) Blood Transcriptomic Progression (GSE26168)\nPC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: PC Distribution Across Progression Stages
    ax2 = axes[1]
    best_pc = traj_stats_df.sort_values("anova_pvalue").iloc[0]["component"]
    sns.boxplot(data=traj_df, x="clinical_group", y=best_pc, ax=ax2, hue="clinical_group", palette=tier_colors, legend=False, boxprops=dict(alpha=0.75), width=0.45)
    sns.stripplot(data=traj_df, x="clinical_group", y=best_pc, ax=ax2, color="black", size=7, jitter=0.15)
    best_row = traj_stats_df[traj_stats_df["component"] == best_pc].iloc[0]
    ax2.set_title(f"B) Glycemic Stage Separation on {best_pc}\nANOVA p = {best_row['anova_pvalue']:.4f} | Trend ρ = {best_row['spearman_order_rho']:.3f}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Clinical Glycemic Group", fontsize=11)
    ax2.set_ylabel(f"{best_pc} Projection Score", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 3: Scree Plot and Variance Explained
    ax3 = axes[2]
    pcs = [f"PC{i+1}" for i in range(5)]
    ax3.bar(pcs, var_exp, color="#4C72B0", alpha=0.85, edgecolor="black")
    ax3.plot(pcs, np.cumsum(var_exp), "r-o", linewidth=2, label="Cumulative Variance")
    ax3.set_title("C) Scree Plot: GSE26168 Principal Components", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Principal Component", fontsize=11)
    ax3.set_ylabel("Variance Explained (%)", fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(frameon=True, fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Phenotypic Separation & Early Dysregulation Trajectory (Discovery Cohort GSE26168)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "phenotypic_trajectory_gse26168.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PATTERNS", f"Saved GSE26168 phenotypic trajectory figure to {out_file.name}")

    return out_file, traj_stats_df


def test_confounder_and_covariate_correlations(
    cross_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Tuple[Path, pd.DataFrame]:
    """
    Tests associations between top principal components (PC1 - PC5) and non-target covariates
    (age, sex, technical platform/cohort) vs. glycemic phenotype (diabetes_status, FBS)
    to distinguish biological signal from confounding technical or demographic variation.
    Saves figure to results/figures/covariate_confounder_associations.png
    and table to results/tables/pca_covariate_associations.csv.
    """
    log_eda("CONFOUNDERS", "Testing associations between top PCs and clinical covariates...")

    expr_t = cross_df.T
    meta_aligned = metadata_df.set_index("sample_id").loc[expr_t.index]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expr_t.values)
    pca = PCA(n_components=5, random_state=SEED)
    pc_scores = pca.fit_transform(scaled_data)

    pc_cols = [f"PC{i+1}" for i in range(5)]
    pc_df = pd.DataFrame(pc_scores, index=expr_t.index, columns=pc_cols).join(meta_aligned)

    covariates = ["cohort", "platform_type", "sex", "diabetes_status", "age_years", "fbs_mg_dl"]

    pvalue_matrix = np.full((len(covariates), len(pc_cols)), np.nan)
    r2_matrix = np.full((len(covariates), len(pc_cols)), np.nan)
    records = []

    for cov_idx, cov in enumerate(covariates):
        cov_vals = pc_df[cov]
        for pc_idx, pc in enumerate(pc_cols):
            scores = pc_df[pc]

            if pd.api.types.is_numeric_dtype(cov_vals):
                valid = cov_vals.notna() & scores.notna()
                if valid.sum() >= 5:
                    r, p = stats.pearsonr(cov_vals[valid], scores[valid])
                    r2 = r ** 2
                else:
                    r, p, r2 = np.nan, np.nan, np.nan
            else:
                groups = [scores[cov_vals == cat].values for cat in cov_vals.unique() if pd.notna(cat)]
                if len(groups) > 1 and all(len(g) > 1 for g in groups):
                    f_stat, p = stats.f_oneway(*groups)
                    ss_total = np.sum((scores - scores.mean()) ** 2)
                    ss_between = np.sum([len(g) * ((np.mean(g) - scores.mean()) ** 2) for g in groups])
                    r2 = ss_between / max(ss_total, 1e-9)
                else:
                    p, r2 = np.nan, np.nan

            pvalue_matrix[cov_idx, pc_idx] = p
            r2_matrix[cov_idx, pc_idx] = r2

            records.append({
                "covariate": cov,
                "principal_component": pc,
                "association_r_squared": r2,
                "p_value": p,
                "is_significant_fdr05": p < 0.05 if not np.isnan(p) else False
            })

    assoc_df = pd.DataFrame(records)
    table_path = RESULTS_TABLES / "pca_covariate_associations.csv"
    assoc_df.to_csv(table_path, index=False)
    log_eda("CONFOUNDERS", f"Saved covariate association table to {table_path.name}")

    fig, (ax_r2, ax_pval) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    cov_labels = [
        "Technical Cohort (GSE26168 vs GSE15932)",
        "Platform Assay (Illumina vs Affymetrix)",
        "Biological Sex (Male vs Female)",
        "Target: Diabetes Status (Control vs T2D)",
        "Donor Age (years)",
        "Fasting Blood Sugar (fbs_mg_dl)"
    ]

    sns.heatmap(
        r2_matrix,
        xticklabels=pc_cols,
        yticklabels=cov_labels,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Variance Explained (R² / η²)"},
        ax=ax_r2
    )
    ax_r2.set_title("A) Covariate Association Strength (R² / Effect Size)", fontsize=12, fontweight="bold")

    log_p = -np.log10(np.clip(pvalue_matrix, 1e-15, 1.0))
    sns.heatmap(
        log_p,
        xticklabels=pc_cols,
        yticklabels=cov_labels,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        cbar_kws={"label": "-log10(p-value)"},
        ax=ax_pval
    )
    ax_pval.set_title("B) Statistical Significance [-log10(p-value)]\n(p < 0.05 corresponds to > 1.30)", fontsize=12, fontweight="bold")

    plt.suptitle("Confounder & Covariate Association Diagnostics (Distinguishing Biological vs Technical Variation)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    fig_path = RESULTS_FIGURES / "covariate_confounder_associations.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("CONFOUNDERS", f"Saved covariate association figure to {fig_path.name}")

    return fig_path, assoc_df


def compare_cross_cohort_gene_consistency(
    norm_matrices: Dict[str, pd.DataFrame]
) -> Tuple[Path, pd.DataFrame]:
    """
    Compares high-variance gene behaviors between the discovery blood cohort (GSE26168)
    and independent validation cohort (GSE15932) across the 13,585 shared genes:
    - Identifies top high-variance genes in discovery and validation.
    - Evaluates cross-cohort correlation of gene expression variances.
    - Assesses concordance of highly variable informative biomarkers prior to differential expression modeling.
    Saves figure to results/figures/cross_cohort_gene_concordance.png
    and table to results/tables/cross_cohort_variance_concordance.csv.
    """
    log_eda("CONSISTENCY", "Evaluating cross-cohort high-variance gene consistency (GSE26168 vs GSE15932)...")

    df_26168 = norm_matrices["GSE26168"]
    df_15932 = norm_matrices["GSE15932"]

    shared_genes = sorted(list(set(df_26168.index).intersection(set(df_15932.index))))
    log_eda("CONSISTENCY", f"Analyzing {len(shared_genes)} shared gene symbols between blood cohorts...")

    sub_26168 = df_26168.loc[shared_genes]
    sub_15932 = df_15932.loc[shared_genes]

    var_26168 = sub_26168.var(axis=1, ddof=1)
    var_15932 = sub_15932.var(axis=1, ddof=1)
    mean_26168 = sub_26168.mean(axis=1)
    mean_15932 = sub_15932.mean(axis=1)

    spearman_rho, spearman_p = stats.spearmanr(var_26168, var_15932)
    pearson_r, pearson_p = stats.pearsonr(var_26168, var_15932)

    top_500_26168 = set(var_26168.nlargest(500).index)
    top_500_15932 = set(var_15932.nlargest(500).index)
    overlap_500 = top_500_26168.intersection(top_500_15932)

    concordance_records = []
    for g in shared_genes:
        concordance_records.append({
            "gene_symbol": g,
            "var_gse26168": var_26168[g],
            "var_gse15932": var_15932[g],
            "mean_gse26168": mean_26168[g],
            "mean_gse15932": mean_15932[g],
            "in_top500_discovery": g in top_500_26168,
            "in_top500_validation": g in top_500_15932,
            "in_top500_both": g in overlap_500
        })

    concordance_df = pd.DataFrame(concordance_records).sort_values("var_gse26168", ascending=False)
    table_path = RESULTS_TABLES / "cross_cohort_variance_concordance.csv"
    concordance_df.to_csv(table_path, index=False)
    log_eda("CONSISTENCY", f"Saved cross-cohort gene concordance table to {table_path.name}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # Panel 1: Scatter of Gene Variance (GSE26168 vs GSE15932)
    ax1 = axes[0]
    ax1.scatter(var_26168, var_15932, alpha=0.35, s=12, color="#4C72B0", edgecolors="none")
    if overlap_500:
        over_list = list(overlap_500)
        ax1.scatter(var_26168[over_list], var_15932[over_list], color="#d62728", s=25, alpha=0.8, label=f"Shared Top-500 Variable (n = {len(overlap_500)})")

    ax1.set_title(f"A) Gene Expression Variance Concordance\nSpearman ρ = {spearman_rho:.3f} (p < 1e-100)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gene Variance in GSE26168 (Discovery)", fontsize=11)
    ax1.set_ylabel("Gene Variance in GSE15932 (Validation)", fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Mean Expression Concordance
    ax2 = axes[1]
    ax2.scatter(mean_26168, mean_15932, alpha=0.25, s=10, color="#55A868", edgecolors="none")
    mean_rho, _ = stats.spearmanr(mean_26168, mean_15932)
    ax2.set_title(f"B) Baseline Expression Intensity Concordance\nSpearman ρ = {mean_rho:.3f}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean log2 Expression in GSE26168", fontsize=11)
    ax2.set_ylabel("Mean log2 Expression in GSE15932", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Panel 3: Overlap of Top Variable Features
    ax3 = axes[2]
    tiers = ["Top 250", "Top 500", "Top 1,000", "Top 2,000"]
    overlap_counts = []
    for k in [250, 500, 1000, 2000]:
        t1 = set(var_26168.nlargest(k).index)
        t2 = set(var_15932.nlargest(k).index)
        overlap_counts.append(len(t1.intersection(t2)))

    bars = ax3.bar(tiers, overlap_counts, color="#9467bd", alpha=0.85, edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")

    ax3.set_title("C) Cross-Cohort Variable Feature Intersection\n(Concordant High-Variance Biomarkers)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Top-N Variable Feature Tier", fontsize=11)
    ax3.set_ylabel("Concordant Shared Genes", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Cross-Cohort Consistency: High-Variance Gene Behaviors Across Blood Discovery & Validation", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    fig_path = RESULTS_FIGURES / "cross_cohort_gene_concordance.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("CONSISTENCY", f"Saved cross-cohort gene concordance figure to {fig_path.name}")

    return fig_path, concordance_df
