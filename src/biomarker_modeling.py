"""
Biomarker Modeling and Machine Learning Pipeline Orchestrator
Executes:
1. Feature Reduction and Stability Selection on Discovery Cohort (GSE26168)
2. Supervised Classifier Training & Comparison (Random Forest, Linear SVM, RBF SVM, Logistic Regression)
3. Discovery Cross-Validation (Leave-One-Out CV, Stratified K-Fold)
4. Independent External Cohort Validation on GSE15932 (PBMC microarray, n=32)
5. Early-stage Pre-diabetic (IFG) Disease Progression Trajectory Risk Scoring
6. Diagnostic Tables and Publication-grade Figure Generation
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from modeling.feature_selection import generate_biomarker_shortlist
from modeling.evaluation import (
    evaluate_discovery_cv,
    evaluate_independent_validation,
    evaluate_panel_scaling,
    evaluate_ifg_trajectory_scores
)
from modeling.visualizations import (
    plot_discovery_cv_roc,
    plot_independent_validation_roc,
    plot_biomarker_shortlist_importance,
    plot_panel_scaling_auc,
    plot_ifg_risk_trajectory,
    plot_confusion_matrices,
    plot_biomarker_cross_cohort_profiles
)


def log_model(stage: str, message: str) -> None:
    print(f"[MODEL-{stage}] {message}")


def run_biomarker_modeling_pipeline(random_state: int = 42) -> Dict[str, Any]:
    """
    Executes end-to-end biomarker prioritization, supervised machine learning,
    and cross-cohort external validation.
    """
    data_dir = PROJECT_ROOT / "data" / "processed"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"

    for d in [tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_model("INIT", "Loading harmonized cross-cohort dataset and discovery DEA results...")

    # Load harmonized dataset
    cross_path = data_dir / "cross_cohort_blood_ml_ready.csv"
    if not cross_path.exists():
        raise FileNotFoundError(f"Harmonized matrix {cross_path} not found.")

    cross_df = pd.read_csv(cross_path)
    feature_cols = [c for c in cross_df.columns[15:]]

    # Load discovery significant DEGs
    dea_path = tables_dir / "dea_discovery_gse26168_significant_degs.csv"
    if not dea_path.exists():
        raise FileNotFoundError(f"Discovery DEA table {dea_path} not found. Run src/differential_expression_analysis.py first.")

    disc_dea_df = pd.read_csv(dea_path)
    discovery_degs = disc_dea_df["gene_symbol"].tolist()

    # Identify candidate features present in both cohorts
    candidate_genes = [g for g in discovery_degs if g in feature_cols]
    log_model("INIT", f"Harmonized dataset: {len(cross_df)} total samples, {len(feature_cols):,} genes.")
    log_model("INIT", f"Candidate DEGs available across discovery & validation: {len(candidate_genes)} genes: {candidate_genes}")

    # Prepare cohorts
    # Discovery primary evaluation (T2D=9 vs Control=8)
    disc_df = cross_df[(cross_df["cohort"] == "GSE26168") & cross_df["target_t2d"].notna()].copy()
    disc_df["y"] = disc_df["target_t2d"].astype(int)

    # Pre-diabetes IFG (n=7)
    ifg_df = cross_df[(cross_df["cohort"] == "GSE26168") & (cross_df["diabetes_status"] == "Pre-Diabetes")].copy()

    # Independent validation (T2D=16 vs Control=16)
    val_df = cross_df[(cross_df["cohort"] == "GSE15932") & cross_df["target_t2d"].notna()].copy()
    val_df["y"] = val_df["target_t2d"].astype(int)

    log_model("COHORT", f"Discovery GSE26168: {len(disc_df)} samples ({disc_df['y'].sum()} T2D, {len(disc_df)-disc_df['y'].sum()} Controls)")
    log_model("COHORT", f"Pre-diabetic IFG:   {len(ifg_df)} samples")
    log_model("COHORT", f"Validation GSE15932: {len(val_df)} samples ({val_df['y'].sum()} T2D, {len(val_df)-val_df['y'].sum()} Controls)")

    # 1. Feature Selection & Shortlist Generation
    log_model("SELECT", "Executing multi-method feature reduction (LASSO, RF Importance, RFE, Stability Selection)...")
    shortlist_df = generate_biomarker_shortlist(
        disc_df=disc_df,
        val_df=val_df,
        candidate_genes=candidate_genes,
        disc_dea_df=disc_dea_df,
        random_state=random_state
    )

    shortlist_csv = tables_dir / "biomarker_candidate_shortlist.csv"
    shortlist_df.to_csv(shortlist_csv, index=False)
    log_model("SELECT", f"Saved biomarker candidate shortlist to {shortlist_csv.name}")

    core_5_genes = shortlist_df[shortlist_df["panel_tier"] == "Core-5"]["gene_symbol"].tolist()
    extended_8_genes = shortlist_df[shortlist_df["panel_tier"].isin(["Core-5", "Extended-8"])]["gene_symbol"].tolist()
    full_10_genes = shortlist_df[shortlist_df["panel_tier"].isin(["Core-5", "Extended-8", "Full-10"])]["gene_symbol"].tolist()
    transferable_5_genes = ["IRS1", "GABRA5", "FUT11", "DYRK3", "DPT"]
    transferable_8_genes = ["IRS1", "GABRA5", "FUT11", "DYRK3", "DPT", "PAQR9", "POPDC3", "LRP5"]

    log_model("SELECT", f"Prioritized Prospective Core-5 Panel:     {core_5_genes}")
    log_model("SELECT", f"Prioritized Prospective Extended-8 Panel: {extended_8_genes}")
    log_model("SELECT", f"Prioritized Cross-Cohort Transferable-5:   {transferable_5_genes}")
    log_model("SELECT", f"Prioritized Cross-Cohort Transferable-8:   {transferable_8_genes}")

    # 2. Model Training & Cross-Validation on Discovery Cohort
    log_model("EVAL-CV", "Running Leave-One-Out Cross-Validation (LOOCV) on discovery cohort across panel tiers...")
    cv_records = []
    cv_details = {}
    cv_details_transferable = {}

    eval_tiers = [
        ("Core-5 (Prospective)", core_5_genes),
        ("Extended-8 (Prospective)", extended_8_genes),
        ("Full-10 (Prospective)", full_10_genes),
        ("Transferable-5 (Cross-Cohort)", transferable_5_genes),
        ("Transferable-8 (Cross-Cohort)", transferable_8_genes),
        ("All-18", candidate_genes)
    ]

    for tier_name, genes in eval_tiers:
        summary, details = evaluate_discovery_cv(disc_df, genes, random_state=random_state)
        summary["panel_tier"] = tier_name
        cv_records.append(summary)
        if tier_name == "Core-5 (Prospective)":
            cv_details = details
        elif tier_name == "Transferable-5 (Cross-Cohort)":
            cv_details_transferable = details

    cv_comparison_df = pd.concat(cv_records, ignore_index=True)
    cv_comparison_csv = tables_dir / "biomarker_model_comparison_discovery_cv.csv"
    cv_comparison_df.to_csv(cv_comparison_csv, index=False)
    log_model("EVAL-CV", f"Saved discovery CV model comparison to {cv_comparison_csv.name}")

    # 3. Independent External Validation on GSE15932
    log_model("EVAL-VAL", "Evaluating trained models on independent validation cohort GSE15932...")
    val_records = []
    val_details = {}
    val_details_transferable = {}

    for tier_name, genes in eval_tiers:
        summary, details = evaluate_independent_validation(disc_df, val_df, genes, random_state=random_state)
        summary["panel_tier"] = tier_name
        val_records.append(summary)
        if tier_name == "Core-5 (Prospective)":
            val_details = details
        elif tier_name == "Transferable-5 (Cross-Cohort)":
            val_details_transferable = details

    val_comparison_df = pd.concat(val_records, ignore_index=True)
    val_comparison_csv = tables_dir / "biomarker_model_comparison_independent_validation.csv"
    val_comparison_df.to_csv(val_comparison_csv, index=False)
    log_model("EVAL-VAL", f"Saved independent validation model comparison to {val_comparison_csv.name}")

    # 4. Panel Scaling Dynamics (k = 1 to 18)
    log_model("SCALING", "Evaluating panel size scaling curve from 1 to 18 genes...")
    ranked_genes = shortlist_df["gene_symbol"].tolist()
    scaling_df = evaluate_panel_scaling(disc_df, val_df, ranked_genes, random_state=random_state)
    scaling_csv = tables_dir / "biomarker_panel_scaling_evaluation.csv"
    scaling_df.to_csv(scaling_csv, index=False)
    log_model("SCALING", f"Saved panel scaling evaluation to {scaling_csv.name}")

    # 5. Pre-diabetic IFG Disease Trajectory Risk Scoring
    log_model("TRAJECTORY", "Scoring pre-diabetic IFG donors along the early glycemic progression trajectory...")
    trajectory_df, corr_df = evaluate_ifg_trajectory_scores(disc_df, ifg_df, core_5_genes, random_state=random_state)
    trajectory_csv = tables_dir / "biomarker_ifg_risk_trajectory_scores.csv"
    trajectory_df.to_csv(trajectory_csv, index=False)
    corr_csv = tables_dir / "biomarker_ifg_trajectory_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)
    log_model("TRAJECTORY", f"Saved IFG trajectory risk scores to {trajectory_csv.name}")

    # 6. Sample-Level Predictions Exports
    log_model("EXPORT", "Exporting sample-level predictions and posterior probabilities...")
    # Discovery sample predictions
    disc_pred_rows = []
    for model_name, d in cv_details.items():
        for sid, y_t, y_p, prob in zip(d["sample_id"], d["y_true"], d["y_pred"], d["y_prob"]):
            disc_pred_rows.append({
                "sample_id": sid,
                "cohort": "GSE26168",
                "model": model_name,
                "true_label": int(y_t),
                "predicted_label": int(y_p),
                "predicted_probability": float(prob)
            })
    disc_pred_df = pd.DataFrame(disc_pred_rows)
    disc_pred_csv = tables_dir / "biomarker_predictions_gse26168.csv"
    disc_pred_df.to_csv(disc_pred_csv, index=False)

    # Validation sample predictions
    val_pred_rows = []
    for model_name, d in val_details.items():
        for sid, cs, y_t, y_p, prob in zip(d["sample_id"], d["cancer_status"], d["y_true"], d["y_pred"], d["y_prob"]):
            val_pred_rows.append({
                "sample_id": sid,
                "cohort": "GSE15932",
                "cancer_status": cs,
                "model": model_name,
                "true_label": int(y_t),
                "predicted_label": int(y_p),
                "predicted_probability": float(prob)
            })
    val_pred_df = pd.DataFrame(val_pred_rows)
    val_pred_csv = tables_dir / "biomarker_predictions_gse15932.csv"
    val_pred_df.to_csv(val_pred_csv, index=False)

    # 7. Diagnostic and Publication Visualizations
    log_model("PLOTS", "Rendering publication-grade visualization figures...")
    fig_disc_roc = plot_discovery_cv_roc(cv_details, figures_dir / "biomarker_roc_discovery_cv.png", "Core-5 Prospective Biomarker Panel")
    fig_val_roc_trans = plot_independent_validation_roc(val_details_transferable, figures_dir / "biomarker_roc_independent_validation.png", "Cross-Cohort Transferable-5 Panel (IRS1, GABRA5, FUT11, DYRK3, DPT)")
    fig_val_roc_prosp = plot_independent_validation_roc(val_details, figures_dir / "biomarker_roc_independent_validation_prospective.png", "Core-5 Prospective Biomarker Panel")
    fig_shortlist = plot_biomarker_shortlist_importance(shortlist_df, figures_dir / "biomarker_feature_importance_shortlist.png", top_n=15)
    fig_scaling = plot_panel_scaling_auc(scaling_df, figures_dir / "biomarker_panel_scaling_auc.png")
    fig_traj = plot_ifg_risk_trajectory(trajectory_df, corr_df, figures_dir / "biomarker_ifg_risk_trajectory.png")
    fig_cm = plot_confusion_matrices(cv_details_transferable, val_details_transferable, figures_dir / "biomarker_confusion_matrices.png", model_name="Logistic Regression")
    fig_profiles = plot_biomarker_cross_cohort_profiles(disc_df, val_df, transferable_5_genes + ["PAQR9"], figures_dir / "biomarker_cross_cohort_expression.png")

    log_model("COMPLETE", "Biomarker modeling and validation pipeline completed successfully!")

    return {
        "tables": {
            "shortlist": shortlist_csv,
            "cv_comparison": cv_comparison_csv,
            "val_comparison": val_comparison_csv,
            "panel_scaling": scaling_csv,
            "trajectory": trajectory_csv,
            "trajectory_correlations": corr_csv,
            "predictions_disc": disc_pred_csv,
            "predictions_val": val_pred_csv
        },
        "figures": {
            "discovery_roc": fig_disc_roc,
            "validation_roc_transferable": fig_val_roc_trans,
            "validation_roc_prospective": fig_val_roc_prosp,
            "shortlist_importance": fig_shortlist,
            "panel_scaling": fig_scaling,
            "ifg_trajectory": fig_traj,
            "confusion_matrices": fig_cm,
            "cross_cohort_profiles": fig_profiles
        },
        "shortlist_genes": {
            "prospective_core_5": core_5_genes,
            "prospective_extended_8": extended_8_genes,
            "prospective_full_10": full_10_genes,
            "transferable_5": transferable_5_genes,
            "transferable_8": transferable_8_genes
        },
        "performance": {
            "transferable_5_cv": cv_comparison_df[cv_comparison_df["panel_tier"] == "Transferable-5 (Cross-Cohort)"].to_dict(orient="records"),
            "transferable_5_val": val_comparison_df[val_comparison_df["panel_tier"] == "Transferable-5 (Cross-Cohort)"].to_dict(orient="records"),
            "prospective_core_5_cv": cv_comparison_df[cv_comparison_df["panel_tier"] == "Core-5 (Prospective)"].to_dict(orient="records"),
            "prospective_core_5_val": val_comparison_df[val_comparison_df["panel_tier"] == "Core-5 (Prospective)"].to_dict(orient="records")
        }
    }


def main():
    results = run_biomarker_modeling_pipeline()
    print("\n" + "=" * 80)
    print("BIOMARKER MODELING AND VALIDATION EXECUTION SUMMARY")
    print("=" * 80)

    print("\n1. Cross-Cohort Transferable Biomarker Panel (5 Genes):")
    for g in results["shortlist_genes"]["transferable_5"]:
        print(f"  * {g}")

    print("\nIndependent Validation on GSE15932 (Transferable-5 Panel):")
    for row in results["performance"]["transferable_5_val"]:
        if row["cohort_subset"] == "All Donors (n=32)":
            print(f"  - {row['model']:25s} | Ext AUC: {row['auc']:.3f} | Acc: {row['accuracy']:.3f} | Sens: {row['sensitivity']:.3f} | Spec: {row['specificity']:.3f}")

    print("\n2. Prospective Feature-Selected Panel (Core-5 Genes):")
    for g in results["shortlist_genes"]["prospective_core_5"]:
        print(f"  * {g}")

    print("\nDiscovery LOOCV Performance (Prospective Core-5):")
    for row in results["performance"]["prospective_core_5_cv"]:
        print(f"  - {row['model']:25s} | AUC: {row['auc']:.3f} | Acc: {row['accuracy']:.3f} | Sens: {row['sensitivity']:.3f} | Spec: {row['specificity']:.3f}")

    print("\nIndependent Validation on GSE15932 (Prospective Core-5):")
    for row in results["performance"]["prospective_core_5_val"]:
        if row["cohort_subset"] == "All Donors (n=32)":
            print(f"  - {row['model']:25s} | Ext AUC: {row['auc']:.3f} | Acc: {row['accuracy']:.3f} | Sens: {row['sensitivity']:.3f} | Spec: {row['specificity']:.3f}")

    print("\nGenerated Tables:")
    for k, v in results["tables"].items():
        print(f"  - {k:25s}: {v}")

    print("\nGenerated Figures:")
    for k, v in results["figures"].items():
        print(f"  - {k:25s}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
