"""
EDA Package
Modular components for Exploratory Data Analysis:
- common: Reproducibility settings, constants, logging, and data loaders
- descriptive_stats: Cohort demographics, distribution profiles, and variance stability
- visualizations: Publication-grade 300 DPI headless diagnostic plots
- pattern_analysis: Disease trajectories, covariate associations, and cross-cohort concordance
"""

from eda.common import (
    SEED,
    ROOT,
    DATA_PROCESSED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    log_eda,
    load_unified_metadata,
    load_expression_matrices,
)

from eda.descriptive_stats import (
    compute_clinical_demographics_summary,
    compute_sample_and_feature_statistics,
    assess_variance_stability,
)

from eda.visualizations import (
    plot_expression_distribution_boxplots,
    plot_expression_distribution_violins,
    plot_density_distributions,
    plot_variance_stability_diagnostics,
    plot_pca_clustering,
    plot_sample_correlation_heatmaps,
)

from eda.pattern_analysis import (
    analyze_phenotypic_trajectory_gse26168,
    test_confounder_and_covariate_correlations,
    compare_cross_cohort_gene_consistency,
)

__all__ = [
    "SEED",
    "ROOT",
    "DATA_PROCESSED",
    "RESULTS_FIGURES",
    "RESULTS_TABLES",
    "log_eda",
    "load_unified_metadata",
    "load_expression_matrices",
    "compute_clinical_demographics_summary",
    "compute_sample_and_feature_statistics",
    "assess_variance_stability",
    "plot_expression_distribution_boxplots",
    "plot_expression_distribution_violins",
    "plot_density_distributions",
    "plot_variance_stability_diagnostics",
    "plot_pca_clustering",
    "plot_sample_correlation_heatmaps",
    "analyze_phenotypic_trajectory_gse26168",
    "test_confounder_and_covariate_correlations",
    "compare_cross_cohort_gene_consistency",
]
