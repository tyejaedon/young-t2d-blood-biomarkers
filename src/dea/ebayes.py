"""
Statistical Differential Expression Engine
Implements:
- Smyth (2004) Empirical Bayes Moderated t-statistics (limma equivalent)
- Two-sample pooled and Welch's t-tests
- Log2 fold change calculations
- Benjamini-Hochberg False Discovery Rate (FDR) adjustments
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma
from scipy.optimize import root_scalar
from statsmodels.stats.multitest import multipletests


def fit_ebayes_priors(
    s2_g: np.ndarray,
    df_g: float
) -> Tuple[float, float]:
    """
    Estimates prior degrees of freedom d0 and prior variance s0^2
    using the method of moments on log-transformed sample variances (Smyth, 2004).
    """
    valid = (s2_g > 1e-12) & np.isfinite(s2_g)
    s2_valid = s2_g[valid]

    if len(s2_valid) < 10:
        return 0.0, float(np.nanmean(s2_valid))

    # e_g = log(s_g^2) - digamma(d_g / 2) + log(d_g / 2)
    e_g = np.log(s2_valid) - digamma(df_g / 2.0) + np.log(df_g / 2.0)
    mean_e = float(np.mean(e_g))
    var_e = float(np.var(e_g, ddof=1))

    trigamma_dg = float(polygamma(1, df_g / 2.0))
    var_diff = var_e - trigamma_dg

    if var_diff > 1e-6:
        try:
            # Solve trigamma(d0 / 2) = var_diff
            res = root_scalar(
                lambda d0: polygamma(1, d0 / 2.0) - var_diff,
                bracket=[1e-4, 1e4],
                method="brentq"
            )
            d0 = float(res.root)
            s0_sq = float(np.exp(mean_e + digamma(d0 / 2.0) - np.log(d0 / 2.0)))
        except Exception:
            d0 = 2.0
            s0_sq = float(np.median(s2_valid))
    else:
        # Variance of sample variances is smaller than expected by chance -> heavy shrinkage
        d0 = 1e4
        s0_sq = float(np.mean(s2_valid))

    return d0, s0_sq


def run_ebayes_dea(
    df_expr: pd.DataFrame,
    case_samples: List[str],
    control_samples: List[str],
    log2fc_threshold: float = 1.0,
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Executes Limma-equivalent Empirical Bayes moderated differential expression analysis:
    - Calculates log2FC = Mean(Case) - Mean(Control)
    - Estimates pooled residual variances s_g^2
    - Shrinks per-gene variances toward empirical Bayes prior s0^2
    - Computes moderated t-statistics and Benjamini-Hochberg FDR
    - Flags statistical significance against strict thresholds
    """
    missing_cases = [s for s in case_samples if s not in df_expr.columns]
    missing_ctrls = [s for s in control_samples if s not in df_expr.columns]
    if missing_cases or missing_ctrls:
        raise ValueError(f"Missing samples in expression matrix: cases={missing_cases}, controls={missing_ctrls}")

    n1 = len(case_samples)
    n2 = len(control_samples)
    d_g = float(n1 + n2 - 2)

    x1 = df_expr[case_samples].values
    x2 = df_expr[control_samples].values

    m1 = np.mean(x1, axis=1)
    m2 = np.mean(x2, axis=1)
    log2fc = m1 - m2

    v1 = np.var(x1, axis=1, ddof=1)
    v2 = np.var(x2, axis=1, ddof=1)

    # Pooled variance across groups
    s2_g = ((n1 - 1) * v1 + (n2 - 1) * v2) / d_g

    # Fit empirical Bayes hyper-parameters
    d0, s0_sq = fit_ebayes_priors(s2_g, d_g)

    # Moderated variance & standard error
    s2_moderated = (d0 * s0_sq + d_g * s2_g) / (d0 + d_g)
    se_moderated = np.sqrt(s2_moderated * (1.0 / n1 + 1.0 / n2))

    # Avoid zero division
    se_moderated = np.where(se_moderated < 1e-12, np.nan, se_moderated)
    t_moderated = log2fc / se_moderated
    df_total = d0 + d_g

    p_moderated = 2.0 * stats.t.sf(np.abs(t_moderated), df=df_total)

    # Ordinary t-test comparison for benchmarking
    se_ordinary = np.sqrt(s2_g * (1.0 / n1 + 1.0 / n2))
    se_ordinary = np.where(se_ordinary < 1e-12, np.nan, se_ordinary)
    t_ordinary = log2fc / se_ordinary
    p_ordinary = 2.0 * stats.t.sf(np.abs(t_ordinary), df=d_g)

    # Build results table
    results = pd.DataFrame({
        "gene_symbol": df_expr.index,
        "log2FC": log2fc,
        "mean_case": m1,
        "mean_control": m2,
        "base_mean": (m1 + m2) / 2.0,
        "raw_variance": s2_g,
        "moderated_variance": s2_moderated,
        "t_ordinary": t_ordinary,
        "p_value_ordinary": p_ordinary,
        "t_moderated": t_moderated,
        "p_value": p_moderated
    }, index=df_expr.index)

    # Benjamini-Hochberg FDR correction on moderated p-values
    valid_mask = results["p_value"].notna()
    results["adj_p_val"] = np.nan
    results["adj_p_val_ordinary"] = np.nan

    if valid_mask.sum() > 0:
        _, padj_mod, _, _ = multipletests(results.loc[valid_mask, "p_value"].values, alpha=fdr_threshold, method="fdr_bh")
        results.loc[valid_mask, "adj_p_val"] = padj_mod

    valid_ord_mask = results["p_value_ordinary"].notna()
    if valid_ord_mask.sum() > 0:
        _, padj_ord, _, _ = multipletests(results.loc[valid_ord_mask, "p_value_ordinary"].values, alpha=fdr_threshold, method="fdr_bh")
        results.loc[valid_ord_mask, "adj_p_val_ordinary"] = padj_ord

    # B-statistic (log-odds of differential expression, empirical Bayes)
    # L = (t_moderated^2) * (d0 / (d0 + d_g)) - approximation
    results["B_stat"] = np.log(np.clip(t_moderated ** 2, 1e-6, None)) - np.log(df_total)

    # Significance classification
    results["is_significant"] = (results["adj_p_val"] < fdr_threshold) & (results["log2FC"].abs() >= log2fc_threshold)
    results["regulation"] = "Not Significant"
    results.loc[results["is_significant"] & (results["log2FC"] > 0), "regulation"] = "Up-regulated"
    results.loc[results["is_significant"] & (results["log2FC"] < 0), "regulation"] = "Down-regulated"

    # Nominal significance (|log2FC| >= 1.0 & raw p < 0.05)
    results["is_nominal"] = (results["p_value"] < 0.05) & (results["log2FC"].abs() >= log2fc_threshold)

    # Sort primarily by statistical significance
    results = results.sort_values(by=["adj_p_val", "p_value"], ascending=True)

    return results


def run_ttest_dea(
    df_expr: pd.DataFrame,
    case_samples: List[str],
    control_samples: List[str],
    equal_var: bool = False,
    log2fc_threshold: float = 1.0,
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Standard two-sample t-test (Student or Welch) for baseline benchmarking.
    """
    x1 = df_expr[case_samples]
    x2 = df_expr[control_samples]

    m1 = x1.mean(axis=1)
    m2 = x2.mean(axis=1)
    log2fc = m1 - m2

    t_stat, p_val = stats.ttest_ind(x1, x2, axis=1, equal_var=equal_var)

    results = pd.DataFrame({
        "gene_symbol": df_expr.index,
        "log2FC": log2fc,
        "mean_case": m1,
        "mean_control": m2,
        "t_stat": t_stat,
        "p_value": p_val
    }, index=df_expr.index)

    valid_mask = results["p_value"].notna()
    results["adj_p_val"] = np.nan
    if valid_mask.sum() > 0:
        _, padj, _, _ = multipletests(results.loc[valid_mask, "p_value"].values, alpha=fdr_threshold, method="fdr_bh")
        results.loc[valid_mask, "adj_p_val"] = padj

    results["is_significant"] = (results["adj_p_val"] < fdr_threshold) & (results["log2FC"].abs() >= log2fc_threshold)
    results["regulation"] = "Not Significant"
    results.loc[results["is_significant"] & (results["log2FC"] > 0), "regulation"] = "Up-regulated"
    results.loc[results["is_significant"] & (results["log2FC"] < 0), "regulation"] = "Down-regulated"

    return results.sort_values(by=["adj_p_val", "p_value"], ascending=True)
