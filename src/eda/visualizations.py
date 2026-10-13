"""
Exploratory Visualizations Suite
Generates high-resolution 300 DPI figures using headless rendering:
- Sample expression distribution boxplots & violins
- Dynamic range Kernel Density Estimation (KDE) and density histograms
- Mean-variance relationship trend diagnostics
- 2D & 3D PCA projection suite colored by phenotype, sex, and platform
- Hierarchical clustering dendrograms and pairwise correlation clustermaps
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from eda.common import (
    SEED,
    RESULTS_FIGURES,
    log_eda
)


def plot_expression_distribution_boxplots(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> Path:
    """
    Creates high-resolution boxplots comparing sample distributions
    pre- vs. post-normalization across all 3 cohorts.
    Confirms quantile equalization and scaling parity across arrays.
    Saves to results/figures/qc_expression_boxplots.png.
    """
    log_eda("PLOT", "Generating pre- vs post-normalization expression boxplots...")

    fig, axes = plt.subplots(3, 2, figsize=(20, 16), dpi=300)
    cohorts = ["GSE26168", "GSE15932", "GSE164416"]
    cohort_labels = {
        "GSE26168": "GSE26168 (Blood Discovery - Illumina)",
        "GSE15932": "GSE15932 (Blood Validation - Affymetrix)",
        "GSE164416": "GSE164416 (Pancreatic Islets - RNA-Seq)"
    }

    for row_idx, cohort in enumerate(cohorts):
        raw_df = raw_matrices[cohort]
        norm_df = norm_matrices[cohort]

        # Raw plot (Left)
        ax_raw = axes[row_idx, 0]
        if cohort == "GSE164416":
            raw_plot_data = [np.log2(raw_df[col].values + 1.0) for col in raw_df.columns]
            raw_title = f"{cohort_labels[cohort]} - Raw Counts [log2(counts + 1)]"
            ylabel = "log2(Raw Counts + 1)"
        elif cohort == "GSE26168":
            raw_plot_data = [raw_df[col].clip(lower=0).values for col in raw_df.columns]
            raw_title = f"{cohort_labels[cohort]} - Raw BeadArray Intensities"
            ylabel = "Raw Intensity"
        else:
            raw_plot_data = [raw_df[col].values for col in raw_df.columns]
            raw_title = f"{cohort_labels[cohort]} - Raw Affymetrix Intensities"
            ylabel = "Raw Intensity"

        ax_raw.boxplot(
            raw_plot_data,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#4C72B0", color="#1c355e", alpha=0.8),
            medianprops=dict(color="#d62728", linewidth=1.5),
            whiskerprops=dict(color="#1c355e"),
            capprops=dict(color="#1c355e")
        )
        ax_raw.set_title(f"PRE-NORMALIZATION: {raw_title}", fontsize=12, fontweight="bold")
        ax_raw.set_xlabel(f"Samples (n = {raw_df.shape[1]})", fontsize=10)
        ax_raw.set_ylabel(ylabel, fontsize=10)
        ax_raw.grid(axis="y", linestyle="--", alpha=0.4)
        ax_raw.set_xticks([])

        # Normalized plot (Right)
        ax_norm = axes[row_idx, 1]
        norm_plot_data = [norm_df[col].values for col in norm_df.columns]
        ax_norm.boxplot(
            norm_plot_data,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#55A868", color="#1b4d27", alpha=0.8),
            medianprops=dict(color="#d62728", linewidth=1.5),
            whiskerprops=dict(color="#1b4d27"),
            capprops=dict(color="#1b4d27")
        )
        norm_title = "Quantile-Normalized log2 Expression" if cohort != "GSE164416" else "DESeq2 / Size-Factor Normalized log2 Counts"
        ax_norm.set_title(f"POST-NORMALIZATION: {cohort_labels[cohort]} - {norm_title}", fontsize=12, fontweight="bold")
        ax_norm.set_xlabel(f"Samples (n = {norm_df.shape[1]})", fontsize=10)
        ax_norm.set_ylabel("Normalized log2 Expression", fontsize=10)
        ax_norm.grid(axis="y", linestyle="--", alpha=0.4)
        ax_norm.set_xticks([])

    plt.suptitle("Quality Control: Sample-Level Expression Distributions Pre- vs. Post-Normalization", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "qc_expression_boxplots.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved boxplots figure to {out_file.name}")
    return out_file


def plot_expression_distribution_violins(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> Path:
    """
    Creates high-resolution violin plots comparing sample distribution shapes
    pre- vs. post-normalization across cohorts.
    Saves to results/figures/qc_expression_violins.png.
    """
    log_eda("PLOT", "Generating pre- vs post-normalization expression violin plots...")

    fig, axes = plt.subplots(3, 2, figsize=(20, 15), dpi=300)
    cohorts = ["GSE26168", "GSE15932", "GSE164416"]

    for row_idx, cohort in enumerate(cohorts):
        raw_df = raw_matrices[cohort]
        norm_df = norm_matrices[cohort]

        # Downsample features for responsive violin rendering (5,000 random features)
        n_features = min(5000, len(raw_df))
        sub_indices = np.random.RandomState(SEED).choice(raw_df.index, size=n_features, replace=False)

        raw_sub = raw_df.loc[sub_indices]
        if cohort == "GSE164416":
            raw_sub = np.log2(raw_sub.clip(lower=0) + 1.0)
        elif cohort in ["GSE26168", "GSE15932"]:
            raw_sub = np.log2(raw_sub.clip(lower=0) + 1.0)

        norm_sub_indices = np.random.RandomState(SEED).choice(norm_df.index, size=min(5000, len(norm_df)), replace=False)
        norm_sub = norm_df.loc[norm_sub_indices]

        # Format long for seaborn violinplot (subset to 12 samples for clean layout)
        selected_samples_raw = list(raw_df.columns[:12])
        raw_melted = raw_sub[selected_samples_raw].melt(var_name="Sample", value_name="Expression")

        selected_samples_norm = list(norm_df.columns[:12])
        norm_melted = norm_sub[selected_samples_norm].melt(var_name="Sample", value_name="Expression")

        # Raw Violin
        ax_raw = axes[row_idx, 0]
        sns.violinplot(
            data=raw_melted,
            x="Sample",
            y="Expression",
            ax=ax_raw,
            color="#4C72B0",
            cut=0,
            density_norm="width",
            inner="quartile",
            linewidth=0.8
        )
        ax_raw.set_title(f"PRE-NORMALIZATION: {cohort} Expression Density Shapes", fontsize=12, fontweight="bold")
        ax_raw.tick_params(axis="x", rotation=45, labelsize=8)
        ax_raw.grid(axis="y", linestyle="--", alpha=0.3)

        # Norm Violin
        ax_norm = axes[row_idx, 1]
        sns.violinplot(
            data=norm_melted,
            x="Sample",
            y="Expression",
            ax=ax_norm,
            color="#55A868",
            cut=0,
            density_norm="width",
            inner="quartile",
            linewidth=0.8
        )
        ax_norm.set_title(f"POST-NORMALIZATION: {cohort} Equalized Density Shapes", fontsize=12, fontweight="bold")
        ax_norm.tick_params(axis="x", rotation=45, labelsize=8)
        ax_norm.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Quality Control: Sample Expression Density Profiles (Pre- vs. Post-Normalization)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "qc_expression_violins.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved violin plots figure to {out_file.name}")
    return out_file


def plot_density_distributions(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> Path:
    """
    Plots Kernel Density Estimation (KDE) and density histograms comparing dynamic ranges
    across samples and cohorts pre- and post-normalization.
    Saves to results/figures/qc_density_distributions.png.
    """
    log_eda("PLOT", "Generating KDE and dynamic range density histograms...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=300)
    cohorts = ["GSE26168", "GSE15932", "GSE164416"]

    for row_idx, cohort in enumerate(cohorts):
        raw_df = raw_matrices[cohort]
        norm_df = norm_matrices[cohort]

        ax_raw = axes[row_idx, 0]
        ax_norm = axes[row_idx, 1]

        # Sample up to 10 representative arrays for KDE curves
        sample_subset = list(raw_df.columns[:10])

        for s in sample_subset:
            # Raw KDE
            r_vals = raw_df[s].values
            if cohort == "GSE164416":
                r_plot = np.log2(r_vals[r_vals > 0] + 1.0)
                xlab_raw = "log2(Counts + 1)"
            else:
                r_plot = np.log2(np.clip(r_vals, 0.1, None) + 1.0)
                xlab_raw = "log2(Intensity + 1)"

            sns.kdeplot(r_plot, ax=ax_raw, linewidth=1.2, alpha=0.6)

            # Norm KDE
            n_vals = norm_df[s].values
            sns.kdeplot(n_vals, ax=ax_norm, linewidth=1.2, alpha=0.6)

        ax_raw.set_title(f"{cohort} - Pre-Normalization KDE Densities", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel(xlab_raw, fontsize=10)
        ax_raw.set_ylabel("Density", fontsize=10)
        ax_raw.grid(True, linestyle="--", alpha=0.3)

        ax_norm.set_title(f"{cohort} - Post-Normalization Equalized Densities", fontsize=11, fontweight="bold")
        ax_norm.set_xlabel("Normalized log2 Expression", fontsize=10)
        ax_norm.set_ylabel("Density", fontsize=10)
        ax_norm.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Exploratory Distribution Diagnostics: Dynamic Range Equalization Across Cohorts", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "qc_density_distributions.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved density distributions to {out_file.name}")
    return out_file


def plot_variance_stability_diagnostics(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> Path:
    """
    Plots mean-variance relationship and dispersion across intensity ranges
    for each cohort pre- vs post-normalization, confirming variance stabilization.
    Saves to results/figures/variance_stability_diagnostics.png.
    """
    log_eda("PLOT", "Generating variance stability diagnostics figure...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), dpi=300)
    cohorts = ["GSE26168", "GSE15932", "GSE164416"]

    for row_idx, cohort in enumerate(cohorts):
        raw_df = raw_matrices[cohort]
        norm_df = norm_matrices[cohort]

        # Raw mean-variance
        ax_raw = axes[row_idx, 0]
        r_vals = np.clip(raw_df.values, 0.0, None)
        r_mean = np.mean(r_vals, axis=1)
        r_std = np.std(r_vals, axis=1, ddof=1)
        r_mask = r_mean > 0.1
        r_m_sub = r_mean[r_mask]
        r_s_sub = r_std[r_mask]

        sub_idx_r = np.random.RandomState(SEED).choice(len(r_m_sub), size=min(5000, len(r_m_sub)), replace=False)
        ax_raw.scatter(np.log2(r_m_sub[sub_idx_r] + 1), np.log2(r_s_sub[sub_idx_r] + 1), alpha=0.25, s=8, color="#4C72B0")

        sns.regplot(
            x=np.log2(r_m_sub[sub_idx_r] + 1),
            y=np.log2(r_s_sub[sub_idx_r] + 1),
            ax=ax_raw,
            scatter=False,
            color="#d62728",
            line_kws={"linewidth": 2}
        )
        rho_r, _ = stats.spearmanr(r_m_sub, r_s_sub)
        ax_raw.set_title(f"{cohort} Pre-Normalization: Mean vs. SD (Spearman ρ = {rho_r:.3f})", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("log2(Mean Intensity + 1)", fontsize=10)
        ax_raw.set_ylabel("log2(Standard Deviation + 1)", fontsize=10)
        ax_raw.grid(True, linestyle="--", alpha=0.3)

        # Norm mean-variance
        ax_norm = axes[row_idx, 1]
        n_vals = norm_df.values
        n_mean = np.mean(n_vals, axis=1)
        n_std = np.std(n_vals, axis=1, ddof=1)
        n_mask = n_mean > 0.01
        n_m_sub = n_mean[n_mask]
        n_s_sub = n_std[n_mask]

        sub_idx_n = np.random.RandomState(SEED).choice(len(n_m_sub), size=min(5000, len(n_m_sub)), replace=False)
        ax_norm.scatter(n_m_sub[sub_idx_n], n_s_sub[sub_idx_n], alpha=0.25, s=8, color="#55A868")

        sns.regplot(
            x=n_m_sub[sub_idx_n],
            y=n_s_sub[sub_idx_n],
            ax=ax_norm,
            scatter=False,
            color="#d62728",
            line_kws={"linewidth": 2}
        )
        rho_n, _ = stats.spearmanr(n_m_sub, n_s_sub)
        ax_norm.set_title(f"{cohort} Post-Normalization: Mean vs. SD (Spearman ρ = {rho_n:.3f})", fontsize=11, fontweight="bold")
        ax_norm.set_xlabel("Normalized Mean log2 Expression", fontsize=10)
        ax_norm.set_ylabel("Standard Deviation", fontsize=10)
        ax_norm.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Variance Stabilization Diagnostics: Mean-Variance Trends Pre- vs. Post-Normalization", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "variance_stability_diagnostics.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved variance stability diagnostics to {out_file.name}")
    return out_file


def plot_pca_clustering(
    cross_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Path:
    """
    Performs 2D and 3D Principal Component Analysis (PCA) projections on harmonized blood samples:
    Colored by:
    - Clinical phenotype (diabetes_status, clinical_group)
    - Biological sex (sex)
    - Platform / Cohort (cohort)
    Screens for batch effects, donor clusters, and evaluates young T2D clustering.
    Saves high-resolution figure to results/figures/eda_pca_clustering.png.
    """
    log_eda("PLOT", "Generating 2D & 3D PCA projection suite...")

    # Transpose so samples are rows, features are columns
    expr_t = cross_df.T
    meta_aligned = metadata_df.set_index("sample_id").loc[expr_t.index]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expr_t.values)

    pca = PCA(n_components=3, random_state=SEED)
    pca_coords = pca.fit_transform(scaled_data)
    var_exp = pca.explained_variance_ratio_ * 100.0

    pca_df = pd.DataFrame(pca_coords, index=expr_t.index, columns=["PC1", "PC2", "PC3"])
    pca_df = pca_df.join(meta_aligned)

    fig = plt.figure(figsize=(20, 16), dpi=300)

    diabetes_palette = {
        "Normoglycemic": "#2ca02c",
        "Pre-Diabetes": "#ff7f0e",
        "Type 2 Diabetes": "#d62728",
        "Unknown": "#7f7f7f"
    }
    sex_palette = {
        "Male": "#1f77b4",
        "Female": "#e377c2",
        "Unspecified": "#7f7f7f",
        "Unknown": "#7f7f7f"
    }
    cohort_palette = {
        "GSE26168": "#9467bd",
        "GSE15932": "#8c564b"
    }

    # Panel 1: 2D PCA by Diabetes Phenotype
    ax1 = fig.add_subplot(2, 2, 1)
    for status, color in diabetes_palette.items():
        sub = pca_df[pca_df["diabetes_status"] == status]
        if len(sub) > 0:
            ax1.scatter(sub["PC1"], sub["PC2"], label=f"{status} (n={len(sub)})", color=color, s=90, alpha=0.85, edgecolors="white", linewidth=1.2)
    ax1.set_title(f"A) Blood Cohort PCA: Glycemic Phenotype\nPC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax1.legend(title="Diabetes Status", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: 2D PCA by Biological Sex
    ax2 = fig.add_subplot(2, 2, 2)
    for sex_val, color in sex_palette.items():
        sub = pca_df[pca_df["sex"] == sex_val]
        if len(sub) > 0:
            ax2.scatter(sub["PC1"], sub["PC2"], label=f"{sex_val} (n={len(sub)})", color=color, s=90, alpha=0.85, edgecolors="white", linewidth=1.2)
    ax2.set_title(f"B) Blood Cohort PCA: Biological Sex\nScreening for Sex-Specific Stratification", fontsize=12, fontweight="bold")
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax2.legend(title="Biological Sex", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Panel 3: 2D PCA by Cohort / Platform
    ax3 = fig.add_subplot(2, 2, 3)
    for cohort_val, color in cohort_palette.items():
        sub = pca_df[pca_df["cohort"] == cohort_val]
        if len(sub) > 0:
            ax3.scatter(sub["PC1"], sub["PC2"], label=f"{cohort_val} (n={len(sub)})", color=color, s=90, alpha=0.85, edgecolors="white", linewidth=1.2)
    ax3.set_title(f"C) Blood Cohort PCA: Technical Platform / Batch Effect\n(GSE26168 Illumina vs. GSE15932 Affymetrix)", fontsize=12, fontweight="bold")
    ax3.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax3.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax3.legend(title="Study Cohort", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Panel 4: 3D PCA Projection
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    for status, color in diabetes_palette.items():
        sub = pca_df[pca_df["diabetes_status"] == status]
        if len(sub) > 0:
            ax4.scatter(sub["PC1"], sub["PC2"], sub["PC3"], label=status, color=color, s=75, alpha=0.85, edgecolors="k", linewidth=0.5)
    ax4.set_title(f"D) 3D Blood Transcriptomic Space\nPC1 ({var_exp[0]:.1f}%) | PC2 ({var_exp[1]:.1f}%) | PC3 ({var_exp[2]:.1f}%)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("PC1", fontsize=10)
    ax4.set_ylabel("PC2", fontsize=10)
    ax4.set_zlabel("PC3", fontsize=10)
    ax4.view_init(elev=22, azim=48)
    ax4.legend(title="Phenotype", frameon=True, fontsize=9)

    plt.suptitle("Dimensionality Reduction: 2D & 3D PCA Projections of Harmonized Blood Cohorts (n = 56 samples)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = RESULTS_FIGURES / "eda_pca_clustering.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved PCA clustering figure to {out_file.name}")
    return out_file


def plot_sample_correlation_heatmaps(
    cross_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Path:
    """
    Computes pairwise sample correlation matrices (Pearson and Spearman)
    with hierarchical clustering and clinical annotation bars to inspect sample-level
    consistency and detect potential outliers.
    Saves to results/figures/sample_correlation_heatmap.png.
    """
    log_eda("PLOT", "Generating pairwise sample correlation heatmaps (Pearson & Spearman)...")

    meta_aligned = metadata_df.set_index("sample_id").loc[cross_df.columns]
    corr_pearson = cross_df.corr(method="pearson")

    status_map = {
        "Normoglycemic": "#2ca02c",
        "Pre-Diabetes": "#ff7f0e",
        "Type 2 Diabetes": "#d62728"
    }
    cohort_map = {
        "GSE26168": "#9467bd",
        "GSE15932": "#8c564b"
    }
    sex_map = {
        "Male": "#1f77b4",
        "Female": "#e377c2"
    }

    col_colors = pd.DataFrame({
        "Phenotype": meta_aligned["diabetes_status"].map(status_map).fillna("#cccccc"),
        "Cohort": meta_aligned["cohort"].map(cohort_map).fillna("#cccccc"),
        "Sex": meta_aligned["sex"].map(sex_map).fillna("#cccccc")
    }, index=cross_df.columns)

    g_pearson = sns.clustermap(
        corr_pearson,
        col_colors=col_colors,
        row_colors=col_colors,
        cmap="vlag",
        vmin=0.80,
        vmax=1.0,
        figsize=(16, 14),
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
        linewidths=0.0
    )
    g_pearson.fig.suptitle("Sample Similarity & Correlation: Pairwise Pearson Clustermap (n = 56 Blood Samples)", fontsize=14, fontweight="bold", y=0.99)

    out_file = RESULTS_FIGURES / "sample_correlation_heatmap.png"
    g_pearson.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    log_eda("PLOT", f"Saved sample correlation clustermap to {out_file.name}")
    return out_file
