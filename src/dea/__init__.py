"""
Differential Expression Analysis (DEA) Package
Implements:
- Smyth (2004) Empirical Bayes Moderated t-statistics (limma-compatible)
- Ordinary Least Squares and two-sample t-tests
- Benjamini-Hochberg False Discovery Rate (FDR) adjustments
- Dual-filtering comparative DEA evaluation
- Publication-grade diagnostic figures (Volcano, MA, Heatmap, Filtering Comparison)
"""

from .ebayes import run_ebayes_dea, run_ttest_dea
from .discovery_analysis import run_discovery_dea_pipeline, compare_filtering_approaches_dea
from .visualizations import (
    plot_volcano,
    plot_ma,
    plot_deg_heatmap,
    plot_pvalue_distribution,
    plot_comparative_deg_yield
)

__all__ = [
    "run_ebayes_dea",
    "run_ttest_dea",
    "run_discovery_dea_pipeline",
    "compare_filtering_approaches_dea",
    "plot_volcano",
    "plot_ma",
    "plot_deg_heatmap",
    "plot_pvalue_distribution",
    "plot_comparative_deg_yield"
]
