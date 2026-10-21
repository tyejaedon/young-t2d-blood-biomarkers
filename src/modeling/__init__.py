"""
Biomarker Modeling and Machine Learning Package
Implements:
- Multi-method feature selection (LASSO, Random Forest Gini Importance, RFE, Stability Selection)
- Supervised classification models (Random Forest, Linear & RBF SVM, Logistic Regression)
- Rigorous cross-validation on discovery cohort (LOOCV, Stratified K-Fold)
- Independent external cohort validation on GSE15932 (PBMC array)
- Pre-diabetic (IFG) disease progression risk scoring
- Publication-grade diagnostic and performance visualizations
"""

from .feature_selection import (
    run_lasso_feature_selection,
    run_random_forest_importance,
    run_rfe_selection,
    run_stability_feature_selection,
    generate_biomarker_shortlist
)
from .classifiers import get_baseline_classifiers, fit_and_predict_model
from .evaluation import (
    evaluate_discovery_cv,
    evaluate_independent_validation,
    evaluate_panel_scaling,
    evaluate_ifg_trajectory_scores
)
from .visualizations import (
    plot_discovery_cv_roc,
    plot_independent_validation_roc,
    plot_biomarker_shortlist_importance,
    plot_panel_scaling_auc,
    plot_ifg_risk_trajectory,
    plot_confusion_matrices,
    plot_biomarker_cross_cohort_profiles
)

__all__ = [
    "run_lasso_feature_selection",
    "run_random_forest_importance",
    "run_rfe_selection",
    "run_stability_feature_selection",
    "generate_biomarker_shortlist",
    "get_baseline_classifiers",
    "fit_and_predict_model",
    "evaluate_discovery_cv",
    "evaluate_independent_validation",
    "evaluate_panel_scaling",
    "evaluate_ifg_trajectory_scores",
    "plot_discovery_cv_roc",
    "plot_independent_validation_roc",
    "plot_biomarker_shortlist_importance",
    "plot_panel_scaling_auc",
    "plot_ifg_risk_trajectory",
    "plot_confusion_matrices",
    "plot_biomarker_cross_cohort_profiles"
]
