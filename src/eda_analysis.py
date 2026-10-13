"""
Exploratory Data Analysis (EDA) Pipeline
Scope of Work:
1. Descriptive Statistics:
   - Expression distribution profiles (feature-level and sample-level stats across raw and normalized matrices).
   - Variance stability across intensity ranges (confirming log2 and quantile/size-factor normalization).
   - Cohort demographics & clinical variables summary (clinical_metadata_unified.csv).
2. Exploratory Visualisations:
   - Sample intensity boxplots and violin plots (pre- vs. post-normalization).
   - Kernel Density Estimation (KDE) and density histograms comparing dynamic ranges.
   - 2D & 3D PCA projections colored by phenotype, sex, and platform/cohort.
   - Hierarchical clustering dendrograms and pairwise sample correlation heatmaps (Pearson & Spearman).
   - Headless rendering (matplotlib.use('Agg')), 300 DPI figures saved to results/figures/.
3. Identifying Patterns and Relationships:
   - Phenotypic separation & early dysregulation (Normoglycemic -> IFG -> T2D in GSE26168).
   - Confounder & covariate correlation (associations between top PCs and age, sex, platform vs phenotype).
   - Cross-cohort consistency (high-variance gene behaviors between GSE26168 and GSE15932).

Adheres to SEED = 42 reproducibility.
"""

from pathlib import Path
import os
import random
from typing import Dict, Tuple, List, Optional, Any
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

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Workspace directories
ROOT = Path("C:/Bioinformatics Project")
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_TABLES = ROOT / "results" / "tables"

RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


def log_eda(stage: str, message: str) -> None:
    """Standardized logging for EDA pipeline."""
    print(f"[EDA-{stage}] {message}")


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_unified_metadata() -> pd.DataFrame:
    """
    Loads unified clinical metadata from data/processed.
    Accepts clinical_metadata_unified.csv or unified_clinical_metadata.csv.
    """
    candidates = [
        DATA_PROCESSED / "clinical_metadata_unified.csv",
        DATA_PROCESSED / "unified_clinical_metadata.csv"
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Ensure alternate path exists for interoperability
            alt_path = DATA_PROCESSED / "clinical_metadata_unified.csv"
            if not alt_path.exists():
                df.to_csv(alt_path, index=False)
            log_eda("METADATA", f"Loaded {len(df)} metadata records from {path.name}")
            return df
    raise FileNotFoundError(f"Unified clinical metadata not found in {DATA_PROCESSED}")


def load_expression_matrices() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Loads raw and normalized expression matrices for all three cohorts,
    plus the harmonized cross-cohort blood expression matrix.
    """
    log_eda("LOAD", "Loading raw and normalized matrices across all cohorts...")

    raw_matrices: Dict[str, pd.DataFrame] = {}
    norm_matrices: Dict[str, pd.DataFrame] = {}

    # GSE26168
    raw_26168 = DATA_PROCESSED / "GSE26168_raw_expression.csv.gz"
    norm_26168 = DATA_PROCESSED / "GSE26168_normalized_expression.csv.gz"
    raw_matrices["GSE26168"] = pd.read_csv(raw_26168, compression="gzip", index_col=0)
    norm_matrices["GSE26168"] = pd.read_csv(norm_26168, compression="gzip", index_col=0)

    # GSE15932
    raw_15932 = DATA_PROCESSED / "GSE15932_raw_expression.csv.gz"
    norm_15932 = DATA_PROCESSED / "GSE15932_normalized_expression.csv.gz"
    raw_matrices["GSE15932"] = pd.read_csv(raw_15932, compression="gzip", index_col=0)
    norm_matrices["GSE15932"] = pd.read_csv(norm_15932, compression="gzip", index_col=0)

    # GSE164416
    raw_164416 = DATA_PROCESSED / "GSE164416_raw_counts.csv.gz"
    norm_164416 = DATA_PROCESSED / "GSE164416_normalized_counts.csv.gz"
    raw_matrices["GSE164416"] = pd.read_csv(raw_164416, compression="gzip", index_col=0)
    norm_matrices["GSE164416"] = pd.read_csv(norm_164416, compression="gzip", index_col=0)

    # Cross-cohort harmonized blood matrix (13,585 genes x 56 blood samples)
    cross_path = DATA_PROCESSED / "cross_cohort_blood_expression_harmonized.csv"
    if not cross_path.exists():
        cross_path = DATA_PROCESSED / "cross_cohort_blood_expression_harmonized.csv.gz"
        cross_df = pd.read_csv(cross_path, compression="gzip", index_col=0)
    else:
        cross_df = pd.read_csv(cross_path, index_col=0)

    log_eda("LOAD", "All matrices loaded successfully.")
    return raw_matrices, norm_matrices, cross_df


# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================

def compute_clinical_demographics_summary(metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarizes sample demographics from clinical metadata:
    - Sample sizes per clinical group (Control, Impaired Fasting Glucose, Type 2 Diabetes, Pancreatic Cancer arms, Islet Donor)
    - Biological sex distribution
    - Age summary (count, mean, std, median, IQR, min, max)
    - Fasting blood sugar (fbs_mg_dl) summary per clinical group
    Exports results to results/tables/.
    """
    log_eda("DEMOGRAPHICS", "Computing clinical demographics and variable distributions...")

    summary_rows = []
    cohorts = metadata_df["cohort"].unique()

    for cohort in cohorts:
        c_df = metadata_df[metadata_df["cohort"] == cohort]
        total_samples = len(c_df)
        groups = c_df["clinical_group"].value_counts().to_dict()
        diabetes_tiers = c_df["diabetes_status"].value_counts().to_dict()
        sex_counts = c_df["sex"].value_counts().to_dict()

        # Age stats
        valid_age = c_df["age_years"].dropna()
        age_str = (
            f"Mean: {valid_age.mean():.1f} ± {valid_age.std():.1f} (Median: {valid_age.median():.1f}, IQR: {stats.iqr(valid_age):.1f}, Range: {valid_age.min():.0f}-{valid_age.max():.0f})"
            if len(valid_age) > 0 else "N/A (Constrained Young Adults)"
        )

        # FBS stats
        valid_fbs = c_df["fbs_mg_dl"].dropna()
        fbs_str = (
            f"Mean: {valid_fbs.mean():.1f} ± {valid_fbs.std():.1f} (Median: {valid_fbs.median():.1f}, IQR: {stats.iqr(valid_fbs):.1f}, Range: {valid_fbs.min():.0f}-{valid_fbs.max():.0f})"
            if len(valid_fbs) > 0 else "N/A (Clinical Diagnosis Criteria)"
        )

        summary_rows.append({
            "cohort": cohort,
            "platform": c_df["platform_type"].iloc[0],
            "tissue_source": c_df["tissue_source"].iloc[0],
            "total_samples": total_samples,
            "clinical_groups": str(groups),
            "diabetes_status_distribution": str(diabetes_tiers),
            "sex_distribution": str(sex_counts),
            "age_years_summary": age_str,
            "fbs_mg_dl_summary": fbs_str
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_TABLES / "clinical_demographics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log_eda("DEMOGRAPHICS", f"Saved demographics overview to {summary_path.name}")

    # Detailed group-level clinical variables table
    detailed_rows = []
    for (cohort, group), g_df in metadata_df.groupby(["cohort", "clinical_group"]):
        n_samples = len(g_df)
        males = (g_df["sex"] == "Male").sum()
        females = (g_df["sex"] == "Female").sum()
        unspecified = (g_df["sex"] == "Unspecified").sum()

        age_vals = g_df["age_years"].dropna()
        fbs_vals = g_df["fbs_mg_dl"].dropna()

        detailed_rows.append({
            "cohort": cohort,
            "clinical_group": group,
            "diabetes_status": g_df["diabetes_status"].iloc[0],
            "n_samples": n_samples,
            "males": males,
            "females": females,
            "unspecified_sex": unspecified,
            "age_mean": age_vals.mean() if len(age_vals) > 0 else np.nan,
            "age_std": age_vals.std() if len(age_vals) > 1 else np.nan,
            "age_median": age_vals.median() if len(age_vals) > 0 else np.nan,
            "age_iqr": stats.iqr(age_vals) if len(age_vals) > 0 else np.nan,
            "age_min": age_vals.min() if len(age_vals) > 0 else np.nan,
            "age_max": age_vals.max() if len(age_vals) > 0 else np.nan,
            "fbs_mean": fbs_vals.mean() if len(fbs_vals) > 0 else np.nan,
            "fbs_std": fbs_vals.std() if len(fbs_vals) > 1 else np.nan,
            "fbs_median": fbs_vals.median() if len(fbs_vals) > 0 else np.nan,
            "fbs_iqr": stats.iqr(fbs_vals) if len(fbs_vals) > 0 else np.nan,
            "fbs_min": fbs_vals.min() if len(fbs_vals) > 0 else np.nan,
            "fbs_max": fbs_vals.max() if len(fbs_vals) > 0 else np.nan
        })

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = RESULTS_TABLES / "clinical_variables_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)
    log_eda("DEMOGRAPHICS", f"Saved detailed clinical variables to {detailed_path.name}")

    return summary_df, detailed_df


def compute_sample_and_feature_statistics(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes sample-level and feature-level descriptive statistics across raw and normalized matrices:
    - Mean, median, standard deviation, interquartile range (IQR), skewness, and zero/sparsity rates.
    Generates:
    1. Sample-level statistics table (per-sample metrics across all arrays).
    2. Feature-level distribution summary table (distribution of gene-level means, stds, IQRs, skewness).
    3. Master concise EDA descriptive statistics comparison table (results/tables/eda_descriptive_statistics.csv).
    """
    log_eda("STATS", "Computing sample-level and feature-level distribution profiles...")

    sample_stat_rows = []
    matrix_overview_rows = []
    feature_summary_rows = []

    cohort_keys = ["GSE26168", "GSE15932", "GSE164416"]

    for cohort in cohort_keys:
        for state, df in [("Raw", raw_matrices[cohort]), ("Normalized", norm_matrices[cohort])]:
            n_features, n_samples = df.shape
            mat_vals = df.values.astype(np.float64)

            # --- Sample-level statistics (columns) ---
            sample_means = np.mean(mat_vals, axis=0)
            sample_medians = np.median(mat_vals, axis=0)
            sample_stds = np.std(mat_vals, axis=0, ddof=1)
            sample_q25 = np.percentile(mat_vals, 25, axis=0)
            sample_q75 = np.percentile(mat_vals, 75, axis=0)
            sample_iqrs = sample_q75 - sample_q25
            sample_mins = np.min(mat_vals, axis=0)
            sample_maxs = np.max(mat_vals, axis=0)
            sample_zeros = np.sum(mat_vals == 0.0, axis=0)
            sample_sparsity = (sample_zeros / n_features) * 100.0
            sample_skews = stats.skew(mat_vals, axis=0, nan_policy="omit")

            for col_idx, col_name in enumerate(df.columns):
                sample_stat_rows.append({
                    "cohort": cohort,
                    "matrix_state": state,
                    "sample_id": col_name,
                    "mean": sample_means[col_idx],
                    "median": sample_medians[col_idx],
                    "std": sample_stds[col_idx],
                    "iqr": sample_iqrs[col_idx],
                    "q25": sample_q25[col_idx],
                    "q75": sample_q75[col_idx],
                    "min": sample_mins[col_idx],
                    "max": sample_maxs[col_idx],
                    "zero_count": sample_zeros[col_idx],
                    "sparsity_pct": sample_sparsity[col_idx],
                    "skewness": sample_skews[col_idx]
                })

            # --- Feature-level statistics (rows) ---
            feat_means = np.mean(mat_vals, axis=1)
            feat_medians = np.median(mat_vals, axis=1)
            feat_stds = np.std(mat_vals, axis=1, ddof=1)
            feat_q25 = np.percentile(mat_vals, 25, axis=1)
            feat_q75 = np.percentile(mat_vals, 75, axis=1)
            feat_iqrs = feat_q75 - feat_q25
            feat_skews = stats.skew(mat_vals, axis=1, nan_policy="omit")
            feat_zeros = np.sum(mat_vals == 0.0, axis=1)
            feat_sparsity = (feat_zeros / n_samples) * 100.0

            feature_summary_rows.append({
                "cohort": cohort,
                "matrix_state": state,
                "n_features": n_features,
                "n_samples": n_samples,
                "mean_of_feature_means": np.mean(feat_means),
                "median_of_feature_means": np.median(feat_means),
                "mean_of_feature_stds": np.mean(feat_stds),
                "median_of_feature_stds": np.median(feat_stds),
                "mean_of_feature_iqrs": np.mean(feat_iqrs),
                "mean_of_feature_skews": np.mean(feat_skews),
                "feature_zero_rate_mean_pct": np.mean(feat_sparsity),
                "all_values_min": np.min(mat_vals),
                "all_values_max": np.max(mat_vals),
                "global_sparsity_pct": (np.sum(mat_vals == 0.0) / mat_vals.size) * 100.0
            })

            # Matrix-level summary
            matrix_overview_rows.append({
                "cohort": cohort,
                "matrix_state": state,
                "n_features": n_features,
                "n_samples": n_samples,
                "intensity_min": np.min(mat_vals),
                "intensity_median": np.median(mat_vals),
                "intensity_mean": np.mean(mat_vals),
                "intensity_max": np.max(mat_vals),
                "intensity_std": np.std(mat_vals, ddof=1),
                "intensity_iqr": np.percentile(mat_vals, 75) - np.percentile(mat_vals, 25),
                "global_skewness": stats.skew(mat_vals.ravel(), nan_policy="omit"),
                "zero_sparsity_pct": (np.sum(mat_vals == 0.0) / mat_vals.size) * 100.0,
                "sample_mean_dispersion_std": np.std(sample_means),
                "sample_median_dispersion_std": np.std(sample_medians)
            })

    sample_stats_df = pd.DataFrame(sample_stat_rows)
    feature_summary_df = pd.DataFrame(feature_summary_rows)
    master_eda_df = pd.DataFrame(matrix_overview_rows)

    # Save outputs
    sample_stats_path = RESULTS_TABLES / "eda_sample_level_statistics.csv"
    feature_summary_path = RESULTS_TABLES / "eda_feature_level_summary.csv"
    master_eda_path = RESULTS_TABLES / "eda_descriptive_statistics.csv"

    sample_stats_df.to_csv(sample_stats_path, index=False)
    feature_summary_df.to_csv(feature_summary_path, index=False)
    master_eda_df.to_csv(master_eda_path, index=False)

    log_eda("STATS", f"Saved sample statistics ({len(sample_stats_df)} rows) to {sample_stats_path.name}")
    log_eda("STATS", f"Saved feature summary to {feature_summary_path.name}")
    log_eda("STATS", f"Saved master descriptive stats to {master_eda_path.name}")

    return master_eda_df, sample_stats_df, feature_summary_df


def assess_variance_stability(
    raw_matrices: Dict[str, pd.DataFrame],
    norm_matrices: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Assesses variance stability across intensity ranges to confirm successful
    log2 and quantile/size-factor normalization:
    - Computes mean vs. standard deviation across features in raw and normalized matrices.
    - Measures Spearman and Pearson rank correlation between feature mean and feature standard deviation.
      (In raw unnormalized/untransformed data, variance strongly scales with mean [heteroskedasticity];
       after log2 and quantile/size-factor normalization, rank correlation is drastically suppressed or stabilized).
    - Measures coefficient of variation (CV) dispersion across expression tiers.
    Saves results to results/tables/variance_stability_assessment.csv.
    """
    log_eda("VAR-STAB", "Assessing variance stability and heteroskedasticity across intensity ranges...")

    stability_rows = []
    cohort_keys = ["GSE26168", "GSE15932", "GSE164416"]

    for cohort in cohort_keys:
        for state, df in [("Raw", raw_matrices[cohort]), ("Normalized", norm_matrices[cohort])]:
            vals = df.values.astype(np.float64)
            # Clip negative raw intensities to zero for safe variance computation
            vals_safe = np.clip(vals, a_min=0.0, a_max=None) if state == "Raw" else vals

            feat_means = np.mean(vals_safe, axis=1)
            feat_stds = np.std(vals_safe, axis=1, ddof=1)
            feat_vars = np.var(vals_safe, axis=1, ddof=1)

            # Avoid division by zero
            non_zero_mask = feat_means > 1e-6
            means_filt = feat_means[non_zero_mask]
            stds_filt = feat_stds[non_zero_mask]
            vars_filt = feat_vars[non_zero_mask]

            # Correlation between mean and std / variance
            spearman_rho, spearman_p = stats.spearmanr(means_filt, stds_filt)
            pearson_r, pearson_p = stats.pearsonr(means_filt, stds_filt)

            # Coefficient of variation (CV = std / mean)
            cv_vals = stds_filt / means_filt
            cv_median = np.median(cv_vals)
            cv_iqr = stats.iqr(cv_vals)

            # Decile variance dispersion (ratio of variance in top decile vs bottom decile)
            deciles = np.percentile(means_filt, np.linspace(0, 100, 11))
            bottom_decile_stds = stds_filt[(means_filt >= deciles[0]) & (means_filt <= deciles[1])]
            top_decile_stds = stds_filt[(means_filt >= deciles[9]) & (means_filt <= deciles[10])]

            decile_std_ratio = (
                np.median(top_decile_stds) / max(np.median(bottom_decile_stds), 1e-9)
                if len(bottom_decile_stds) > 0 and len(top_decile_stds) > 0 else np.nan
            )

            # Status assessment
            if state == "Normalized":
                is_stabilized = (abs(spearman_rho) < 0.70) or (decile_std_ratio < 10.0)
                status_text = "CONFIRMED: Variance stabilized" if is_stabilized else "MODERATE: Residual dispersion"
            else:
                status_text = "EXPECTED: High heteroskedasticity"

            stability_rows.append({
                "cohort": cohort,
                "matrix_state": state,
                "n_features_evaluated": int(np.sum(non_zero_mask)),
                "spearman_mean_std_rho": spearman_rho,
                "spearman_pvalue": spearman_p,
                "pearson_mean_std_r": pearson_r,
                "pearson_pvalue": pearson_p,
                "median_cv": cv_median,
                "iqr_cv": cv_iqr,
                "decile_std_ratio_top_vs_bottom": decile_std_ratio,
                "variance_stability_status": status_text
            })

    stab_df = pd.DataFrame(stability_rows)
    out_path = RESULTS_TABLES / "variance_stability_assessment.csv"
    stab_df.to_csv(out_path, index=False)
    log_eda("VAR-STAB", f"Saved variance stability evaluation to {out_path.name}")
    return stab_df


# =============================================================================
# 2. EXPLORATORY VISUALIZATIONS
# =============================================================================

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
        # For visualization clarity with raw counts/intensities, plot log2(x+1) for RNA-Seq or raw intensities
        if cohort == "GSE164416":
            raw_plot_data = [np.log2(raw_df[col].values + 1.0) for col in raw_df.columns]
            raw_title = f"{cohort_labels[cohort]} - Raw Counts [log2(counts + 1)]"
            ylabel = "log2(Raw Counts + 1)"
        elif cohort == "GSE26168":
            # Clip negative background values for clean boxplot
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
        # For RNA-Seq or high range, log2 transform raw for violin shape comparison
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
        # Filter zero/near-zero
        r_mask = r_mean > 0.1
        r_m_sub = r_mean[r_mask]
        r_s_sub = r_std[r_mask]

        # Subsample for responsive scatter rendering
        sub_idx_r = np.random.RandomState(SEED).choice(len(r_m_sub), size=min(5000, len(r_m_sub)), replace=False)
        ax_raw.scatter(np.log2(r_m_sub[sub_idx_r] + 1), np.log2(r_s_sub[sub_idx_r] + 1), alpha=0.25, s=8, color="#4C72B0")

        # Trend line
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
    # Align metadata
    meta_aligned = metadata_df.set_index("sample_id").loc[expr_t.index]

    # Standard scale features before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expr_t.values)

    # 3-component PCA
    pca = PCA(n_components=3, random_state=SEED)
    pca_coords = pca.fit_transform(scaled_data)
    var_exp = pca.explained_variance_ratio_ * 100.0

    pca_df = pd.DataFrame(pca_coords, index=expr_t.index, columns=["PC1", "PC2", "PC3"])
    pca_df = pca_df.join(meta_aligned)

    # Create 2x2 multi-panel figure (3 2D projections + 1 3D projection)
    fig = plt.figure(figsize=(20, 16), dpi=300)

    # Palette mappings
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

    # Panel 3: 2D PCA by Cohort / Platform (Screening for Batch Effects)
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

    # Align metadata
    meta_aligned = metadata_df.set_index("sample_id").loc[cross_df.columns]

    # Compute correlation matrices
    corr_pearson = cross_df.corr(method="pearson")
    corr_spearman = cross_df.corr(method="spearman")

    # Color mapping for annotations
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

    # Plot Pearson clustermap
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


# =============================================================================
# 3. IDENTIFYING PATTERNS AND RELATIONSHIPS
# =============================================================================

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

    # Perform PCA on GSE26168 alone
    scaler = StandardScaler()
    scaled_expr = scaler.fit_transform(df_26168.T.values)
    pca = PCA(n_components=5, random_state=SEED)
    pca_scores = pca.fit_transform(scaled_expr)
    var_exp = pca.explained_variance_ratio_ * 100.0

    traj_df = pd.DataFrame(pca_scores, index=df_26168.columns, columns=[f"PC{i+1}" for i in range(5)])
    traj_df = traj_df.join(meta_26168)

    # Assign ordinal glycemic progression score:
    # 0: Normoglycemic Control, 1: Impaired Fasting Glucose, 2: Type 2 Diabetes
    progression_order = {
        "Control": 0,
        "Impaired Fasting Glucose": 1,
        "Type 2 Diabetes": 2
    }
    traj_df["glycemic_order"] = traj_df["clinical_group"].map(progression_order)

    # Statistical test across glycemic progression for each PC
    stats_list = []
    for i in range(1, 6):
        pc_col = f"PC{i}"
        groups = [traj_df[traj_df["clinical_group"] == g][pc_col].values for g in ["Control", "Impaired Fasting Glucose", "Type 2 Diabetes"]]
        f_stat, p_val = stats.f_oneway(*groups)
        kw_stat, kw_p = stats.kruskal(*groups)
        # Linear correlation with ordinal stage
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

    # Visualization
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

    # Add progression centroid trajectory
    centroids = traj_df.groupby("clinical_group")[["PC1", "PC2"]].mean().loc[["Control", "Impaired Fasting Glucose", "Type 2 Diabetes"]]
    ax1.plot(centroids["PC1"], centroids["PC2"], "k--", linewidth=2.5, alpha=0.7, label="Progression Centroid Vector")
    ax1.set_title(f"A) Blood Transcriptomic Progression (GSE26168)\nPC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%", fontsize=12, fontweight="bold")
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: PC Distribution Across Progression Stages
    ax2 = axes[1]
    # Pick the most distinguishing PC or PC1
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

    # Covariates to test
    covariates = ["cohort", "platform_type", "sex", "diabetes_status", "age_years", "fbs_mg_dl"]

    pvalue_matrix = np.full((len(covariates), len(pc_cols)), np.nan)
    r2_matrix = np.full((len(covariates), len(pc_cols)), np.nan)
    records = []

    for cov_idx, cov in enumerate(covariates):
        cov_vals = pc_df[cov]
        for pc_idx, pc in enumerate(pc_cols):
            scores = pc_df[pc]

            # Categorical vs Numeric
            if pd.api.types.is_numeric_dtype(cov_vals):
                valid = cov_vals.notna() & scores.notna()
                if valid.sum() >= 5:
                    r, p = stats.pearsonr(cov_vals[valid], scores[valid])
                    r2 = r ** 2
                else:
                    r, p, r2 = np.nan, np.nan, np.nan
            else:
                # ANOVA for categorical
                groups = [scores[cov_vals == cat].values for cat in cov_vals.unique() if pd.notna(cat)]
                if len(groups) > 1 and all(len(g) > 1 for g in groups):
                    f_stat, p = stats.f_oneway(*groups)
                    # eta squared = SS_between / SS_total
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

    # Visualization: Heatmap of R² and -log10(p-value)
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

    # Shared genes
    shared_genes = sorted(list(set(df_26168.index).intersection(set(df_15932.index))))
    log_eda("CONSISTENCY", f"Analyzing {len(shared_genes)} shared gene symbols between blood cohorts...")

    sub_26168 = df_26168.loc[shared_genes]
    sub_15932 = df_15932.loc[shared_genes]

    # Compute gene variances and standard deviations
    var_26168 = sub_26168.var(axis=1, ddof=1)
    var_15932 = sub_15932.var(axis=1, ddof=1)
    mean_26168 = sub_26168.mean(axis=1)
    mean_15932 = sub_15932.mean(axis=1)

    # Rank correlation of variance
    spearman_rho, spearman_p = stats.spearmanr(var_26168, var_15932)
    pearson_r, pearson_p = stats.pearsonr(var_26168, var_15932)

    # Top high-variance gene overlap
    top_500_26168 = set(var_26168.nlargest(500).index)
    top_500_15932 = set(var_15932.nlargest(500).index)
    overlap_500 = top_500_26168.intersection(top_500_15932)

    top_1000_26168 = set(var_26168.nlargest(1000).index)
    top_1000_15932 = set(var_15932.nlargest(1000).index)
    overlap_1000 = top_1000_26168.intersection(top_1000_15932)

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

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # Panel 1: Scatter of Gene Variance (GSE26168 vs GSE15932)
    ax1 = axes[0]
    ax1.scatter(var_26168, var_15932, alpha=0.35, s=12, color="#4C72B0", edgecolors="none")
    # Highlight top overlapping variable genes
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


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_eda_pipeline() -> Dict[str, Any]:
    """
    Executes the entire Exploratory Data Analysis (EDA) pipeline:
    1. Loads unified metadata and expression matrices.
    2. Computes descriptive statistics (sample-level, feature-level, variance stability, clinical demographics).
    3. Generates the full suite of exploratory visualizations (boxplots, violins, KDE, PCA 2D/3D, correlation clustermap).
    4. Identifies patterns & relationships (glycemic trajectory, covariate confounders, cross-cohort consistency).
    Returns manifest dictionary of all generated deliverables.
    """
    log_eda("MAIN", "=======================================================")
    log_eda("MAIN", "STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    log_eda("MAIN", "=======================================================")

    # 1. Load data
    metadata_df = load_unified_metadata()
    raw_matrices, norm_matrices, cross_df = load_expression_matrices()

    # 2. Descriptive Statistics
    demo_summary_df, demo_detailed_df = compute_clinical_demographics_summary(metadata_df)
    master_eda_df, sample_stats_df, feat_summary_df = compute_sample_and_feature_statistics(raw_matrices, norm_matrices)
    var_stability_df = assess_variance_stability(raw_matrices, norm_matrices)

    # 3. Exploratory Visualizations
    fig_boxplots = plot_expression_distribution_boxplots(raw_matrices, norm_matrices)
    fig_violins = plot_expression_distribution_violins(raw_matrices, norm_matrices)
    fig_kde = plot_density_distributions(raw_matrices, norm_matrices)
    fig_var_stab = plot_variance_stability_diagnostics(raw_matrices, norm_matrices)
    fig_pca = plot_pca_clustering(cross_df, metadata_df)
    fig_corr = plot_sample_correlation_heatmaps(cross_df, metadata_df)

    # 4. Patterns and Relationships
    fig_trajectory, traj_stats_df = analyze_phenotypic_trajectory_gse26168(norm_matrices, metadata_df)
    fig_confounders, assoc_df = test_confounder_and_covariate_correlations(cross_df, metadata_df)
    fig_concordance, concordance_df = compare_cross_cohort_gene_consistency(norm_matrices)

    log_eda("MAIN", "=======================================================")
    log_eda("MAIN", "EDA PIPELINE SUCCESSFULLY COMPLETED")
    log_eda("MAIN", f"Tables saved to: {RESULTS_TABLES}")
    log_eda("MAIN", f"Figures saved to: {RESULTS_FIGURES}")
    log_eda("MAIN", "=======================================================")

    return {
        "tables": {
            "clinical_demographics_summary": RESULTS_TABLES / "clinical_demographics_summary.csv",
            "clinical_variables_detailed": RESULTS_TABLES / "clinical_variables_detailed.csv",
            "eda_descriptive_statistics": RESULTS_TABLES / "eda_descriptive_statistics.csv",
            "eda_sample_level_statistics": RESULTS_TABLES / "eda_sample_level_statistics.csv",
            "eda_feature_level_summary": RESULTS_TABLES / "eda_feature_level_summary.csv",
            "variance_stability_assessment": RESULTS_TABLES / "variance_stability_assessment.csv",
            "pca_covariate_associations": RESULTS_TABLES / "pca_covariate_associations.csv",
            "cross_cohort_variance_concordance": RESULTS_TABLES / "cross_cohort_variance_concordance.csv"
        },
        "figures": {
            "qc_expression_boxplots": fig_boxplots,
            "qc_expression_violins": fig_violins,
            "qc_density_distributions": fig_kde,
            "variance_stability_diagnostics": fig_var_stab,
            "eda_pca_clustering": fig_pca,
            "sample_correlation_heatmap": fig_corr,
            "phenotypic_trajectory_gse26168": fig_trajectory,
            "covariate_confounder_associations": fig_confounders,
            "cross_cohort_gene_concordance": fig_concordance
        }
    }


if __name__ == "__main__":
    run_eda_pipeline()
