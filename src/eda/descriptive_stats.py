"""
Descriptive Statistics & Variance Stability Module
Computes:
- Cohort demographics & clinical variables summaries
- Sample-level and feature-level expression distribution profiles
- Heteroskedasticity and variance stability assessment across intensity ranges
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

from eda.common import (
    RESULTS_TABLES,
    log_eda
)


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
