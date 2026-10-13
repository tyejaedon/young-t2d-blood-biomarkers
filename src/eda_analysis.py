"""
Exploratory Data Analysis (EDA) Pipeline Orchestrator
Coordinates:
- Cohort Demographics & Clinical Variables (eda.descriptive_stats)
- Descriptive Statistics & Variance Stability (eda.descriptive_stats)
- Publication-Grade Diagnostic Visualizations (eda.visualizations)
- Phenotypic Trajectory & Confounder Dissection (eda.pattern_analysis)
- Cross-Cohort High-Variance Gene Concordance (eda.pattern_analysis)

Adheres strictly to SEED = 42 reproducibility and headless plotting standards.
"""

from typing import Dict, Any

from eda import (
    SEED,
    ROOT,
    DATA_PROCESSED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    log_eda,
    load_unified_metadata,
    load_expression_matrices,
    compute_clinical_demographics_summary,
    compute_sample_and_feature_statistics,
    assess_variance_stability,
    plot_expression_distribution_boxplots,
    plot_expression_distribution_violins,
    plot_density_distributions,
    plot_variance_stability_diagnostics,
    plot_pca_clustering,
    plot_sample_correlation_heatmaps,
    analyze_phenotypic_trajectory_gse26168,
    test_confounder_and_covariate_correlations,
    compare_cross_cohort_gene_consistency,
)


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
