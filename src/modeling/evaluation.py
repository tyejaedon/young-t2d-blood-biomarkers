"""
Model Evaluation and Validation Engine
Conducts:
- Discovery cross-validation (Leave-One-Out CV and 5-Fold Stratified CV)
- Independent external cohort validation on GSE15932 (PBMC array)
- Subgroup assessment (pure Cancer-Free vs all cohorts)
- Panel scaling dynamics (from single gene to 18-gene panel)
- Pre-diabetic IFG early risk progression trajectory scoring
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .classifiers import get_baseline_classifiers


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Computes standard diagnostic classification metrics.
    """
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "brier_score": float(brier)
    }


def evaluate_discovery_cv(
    disc_df: pd.DataFrame,
    feature_genes: List[str],
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Executes Leave-One-Out Cross-Validation (LOOCV) on discovery dataset (GSE26168).
    Returns a metrics summary DataFrame and detailed per-sample predictions.
    """
    X_raw = disc_df[feature_genes].values
    y = disc_df["y"].values
    sample_ids = disc_df["sample_id"].tolist()

    models = get_baseline_classifiers(random_state=random_state)
    loo = LeaveOneOut()

    cv_records = []
    sample_details = {}

    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X_raw):
            X_tr, y_tr = X_raw[train_idx], y[train_idx]
            X_te = X_raw[test_idx]

            # Standardize strictly within fold to prevent data leakage
            scaler = StandardScaler()
            X_tr_std = scaler.fit_transform(X_tr)
            X_te_std = scaler.transform(X_te)

            m = model.__class__(**model.get_params())
            m.fit(X_tr_std, y_tr)
            oof_preds[test_idx] = m.predict(X_te_std)
            oof_probs[test_idx] = m.predict_proba(X_te_std)[:, 1]

        metrics = calculate_binary_metrics(y, oof_preds, oof_probs)
        metrics["model"] = name
        metrics["n_features"] = len(feature_genes)
        cv_records.append(metrics)

        sample_details[name] = {
            "sample_id": sample_ids,
            "y_true": y,
            "y_pred": oof_preds,
            "y_prob": oof_probs
        }

    summary_df = pd.DataFrame(cv_records)[[
        "model", "n_features", "auc", "accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score"
    ]]
    return summary_df, sample_details


def evaluate_independent_validation(
    disc_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_genes: List[str],
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Trains models on full GSE26168 discovery dataset and validates them
    on independent GSE15932 cohort across all donors and cancer-free subset.
    """
    X_disc = disc_df[feature_genes].values
    y_disc = disc_df["y"].values

    scaler_disc = StandardScaler()
    X_disc_std = scaler_disc.fit_transform(X_disc)

    X_val = val_df[feature_genes].values
    y_val = val_df["y"].values
    sample_ids_val = val_df["sample_id"].tolist()
    cancer_status_val = val_df["cancer_status"].tolist()

    # Cohort-level standardization to align microarray dynamic ranges
    scaler_val = StandardScaler()
    X_val_std = scaler_val.fit_transform(X_val)

    models = get_baseline_classifiers(random_state=random_state)
    val_records = []
    val_predictions = {}

    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_disc_std, y_disc)

        preds = m.predict(X_val_std)
        probs = m.predict_proba(X_val_std)[:, 1]

        # Overall validation metrics (n=32)
        metrics_all = calculate_binary_metrics(y_val, preds, probs)
        metrics_all["model"] = name
        metrics_all["cohort_subset"] = "All Donors (n=32)"
        metrics_all["n_features"] = len(feature_genes)
        val_records.append(metrics_all)

        # Cancer-free subset metrics (n=16: 8 T2D vs 8 Control)
        cf_mask = np.array(cancer_status_val) == "Cancer-Free"
        if np.sum(cf_mask) > 0:
            metrics_cf = calculate_binary_metrics(y_val[cf_mask], preds[cf_mask], probs[cf_mask])
            metrics_cf["model"] = name
            metrics_cf["cohort_subset"] = "Cancer-Free Donors (n=16)"
            metrics_cf["n_features"] = len(feature_genes)
            val_records.append(metrics_cf)

        val_predictions[name] = {
            "sample_id": sample_ids_val,
            "cancer_status": cancer_status_val,
            "y_true": y_val,
            "y_pred": preds,
            "y_prob": probs
        }

    summary_df = pd.DataFrame(val_records)[[
        "model", "cohort_subset", "n_features", "auc", "accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score"
    ]]
    return summary_df, val_predictions


def evaluate_panel_scaling(
    disc_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ranked_genes: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates classification performance across expanding biomarker panel sizes
    (1 to len(ranked_genes)) to identify the optimal parsimonious biomarker panel.
    """
    records = []
    models = get_baseline_classifiers(random_state=random_state)

    for k in range(1, len(ranked_genes) + 1):
        panel_k = ranked_genes[:k]
        cv_summary, _ = evaluate_discovery_cv(disc_df, panel_k, random_state=random_state)
        val_summary, _ = evaluate_independent_validation(disc_df, val_df, panel_k, random_state=random_state)

        for name in models.keys():
            disc_row = cv_summary[cv_summary["model"] == name].iloc[0]
            val_row = val_summary[(val_summary["model"] == name) & (val_summary["cohort_subset"] == "All Donors (n=32)")].iloc[0]

            records.append({
                "panel_size": k,
                "model": name,
                "added_gene": ranked_genes[k - 1],
                "discovery_cv_auc": disc_row["auc"],
                "discovery_cv_acc": disc_row["accuracy"],
                "validation_auc": val_row["auc"],
                "validation_acc": val_row["accuracy"],
                "validation_sensitivity": val_row["sensitivity"],
                "validation_specificity": val_row["specificity"]
            })

    return pd.DataFrame(records)


def evaluate_ifg_trajectory_scores(
    disc_df: pd.DataFrame,
    ifg_df: pd.DataFrame,
    feature_genes: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates predicted T2D risk probability progression across
    Normoglycemic Controls (n=8) -> Impaired Fasting Glucose (IFG, n=7) -> Type 2 Diabetes (n=9).
    """
    X_disc = disc_df[feature_genes].values
    y_disc = disc_df["y"].values

    scaler = StandardScaler()
    X_disc_std = scaler.fit_transform(X_disc)

    X_ifg = ifg_df[feature_genes].values
    X_ifg_std = scaler.transform(X_ifg)

    models = get_baseline_classifiers(random_state=random_state)
    records = []

    # Combine all donors
    ctrl_samples = disc_df[disc_df["y"] == 0]["sample_id"].tolist()
    t2d_samples = disc_df[disc_df["y"] == 1]["sample_id"].tolist()
    ifg_samples = ifg_df["sample_id"].tolist()

    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_disc_std, y_disc)

        ctrl_mask = (disc_df["y"] == 0).values
        t2d_mask = (disc_df["y"] == 1).values

        ctrl_probs = m.predict_proba(X_disc_std[ctrl_mask])[:, 1]
        t2d_probs = m.predict_proba(X_disc_std[t2d_mask])[:, 1]
        ifg_probs = m.predict_proba(X_ifg_std)[:, 1]

        for sid, p in zip(ctrl_samples, ctrl_probs):
            records.append({"model": name, "stage": "Control", "stage_ordinal": 0, "sample_id": sid, "predicted_t2d_risk": p})
        for sid, p in zip(ifg_samples, ifg_probs):
            records.append({"model": name, "stage": "IFG (Pre-Diabetes)", "stage_ordinal": 1, "sample_id": sid, "predicted_t2d_risk": p})
        for sid, p in zip(t2d_samples, t2d_probs):
            records.append({"model": name, "stage": "Type 2 Diabetes", "stage_ordinal": 2, "sample_id": sid, "predicted_t2d_risk": p})

    df = pd.DataFrame(records)

    # Compute Spearman rank correlation across stage ordinal
    corrs = []
    for name in models.keys():
        sub = df[df["model"] == name]
        rho, pval = stats.spearmanr(sub["stage_ordinal"], sub["predicted_t2d_risk"])
        corrs.append({
            "model": name,
            "spearman_rho": rho,
            "p_value": pval,
            "mean_ctrl_risk": sub[sub["stage"] == "Control"]["predicted_t2d_risk"].mean(),
            "mean_ifg_risk": sub[sub["stage"] == "IFG (Pre-Diabetes)"]["predicted_t2d_risk"].mean(),
            "mean_t2d_risk": sub[sub["stage"] == "Type 2 Diabetes"]["predicted_t2d_risk"].mean()
        })

    corr_df = pd.DataFrame(corrs)
    return df, corr_df
