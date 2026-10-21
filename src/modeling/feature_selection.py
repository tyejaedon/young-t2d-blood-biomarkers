"""
Feature Selection Module
Implements:
- L1-penalized Logistic Regression (LASSO) feature reduction
- Random Forest Gini impurity feature importance ranking
- Recursive Feature Elimination (RFE)
- Multi-resample Stability Selection (frequency across cross-validation folds)
- Biomarker shortlist synthesis with functional biological annotations
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Biological pathomechanism dictionary for candidate genes
BIOLOGICAL_ANNOTATIONS = {
    "IRS1": "Insulin Receptor Substrate 1; central transducer of insulin/IGF-1 signaling; serine phosphorylation/downregulation drives peripheral insulin resistance",
    "GABRA5": "Gamma-Aminobutyric Acid Type A Receptor Subunit Alpha5; neuro-endocrine and peripheral leukocyte immunomodulatory channel",
    "FUT11": "Fucosyltransferase 11; alpha-(1,3)-fucosyltransferase governing cell surface glycan remodeling and immune cell trafficking",
    "DYRK3": "Dual-Specificity Tyrosine Phosphorylation Regulated Kinase 3; controls stress granule dynamics and mTORC1 reactivation under nutrient stress",
    "DPT": "Dermatopontin; extracellular matrix protein modulating TGF-beta signaling and tissue remodeling in metabolic syndrome",
    "PAQR9": "Progestin & AdipoQ Receptor Family Member 9; membrane receptor belonging to adiponectin superfamily; modulates lipid oxidation and glucose disposal",
    "POPDC3": "Popeye Domain Containing 3; cAMP-binding membrane effector maintaining cellular and cardiac membrane structural integrity",
    "LRP5": "LDL Receptor-Related Protein 5; Wnt co-receptor critical for pancreatic beta-cell proliferation and glucose-stimulated insulin secretion",
    "ANGPTL2": "Angiopoietin-Like Protein 2; pro-inflammatory circulating adipokine inducing chronic vascular inflammation and peripheral insulin resistance",
    "MALAT1": "Metastasis-Associated Lung Adenocarcinoma Transcript 1; lncRNA mediator of glucose-induced endothelial and vascular inflammation",
    "SYNPO": "Synaptopodin; actin-associated protein maintaining cellular architectural stability under metabolic strain",
    "KISS1": "Kisspeptin-1; neuro-endocrine GPCR peptide ligand regulating whole-body energy homeostasis and metabolic adaptation",
    "MBNL3": "Muscleblind-Like Splicing Regulator 3; alternative splicing factor governing developmental insulin receptor isoform reprogramming",
    "BRD4": "Bromodomain Containing 4; epigenetic reader governing transcriptional activation of pro-inflammatory cytokines and NF-kB signaling",
    "ZNF268": "Zinc Finger Protein 268; transcriptional repressor modulating leukocyte differentiation and stress response",
    "OR4D2": "Olfactory Receptor Family 4 Subfamily D Member 2; ectopic metabolic chemosensory GPCR in circulating immune cells",
    "IFNA7": "Interferon Alpha 7; innate type I interferon cytokine orchestrating antiviral and sterile inflammatory responses",
    "C3orf27": "Chromosome 3 Open Reading Frame 27; dysregulated leukocyte transcriptomic feature associated with metabolic dysfunction",
    "AKT3": "RAC-gamma Serine/Threonine Kinase; downstream effector of IRS1/PI3K pathway mediating glucose uptake and beta-cell survival",
    "APOL4": "Apolipoprotein L4; lipid-binding protein implicated in intracellular lipid transport and innate immunity"
}


def run_lasso_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    c_param: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Applies L1-penalized Logistic Regression (LASSO) to enforce sparsity
    and extract non-zero predictive features.
    """
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=c_param, random_state=random_state)
    clf.fit(X, y)
    coefs = clf.coef_[0]

    df = pd.DataFrame({
        "gene_symbol": feature_names,
        "lasso_coefficient": coefs,
        "lasso_abs_coef": np.abs(coefs),
        "lasso_selected": coefs != 0
    }).sort_values("lasso_abs_coef", ascending=False).reset_index(drop=True)

    df["lasso_rank"] = df["lasso_abs_coef"].rank(ascending=False, method="min").astype(int)
    return df


def run_random_forest_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes Gini impurity-based feature importances using Random Forest.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=3, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_

    df = pd.DataFrame({
        "gene_symbol": feature_names,
        "rf_importance": importances
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

    df["rf_rank"] = df["rf_importance"].rank(ascending=False, method="min").astype(int)
    return df


def run_rfe_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features_to_select: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs Recursive Feature Elimination (RFE) using Logistic Regression.
    """
    base_estimator = LogisticRegression(C=1.0, random_state=random_state)
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)

    df = pd.DataFrame({
        "gene_symbol": feature_names,
        "rfe_selected": rfe.support_,
        "rfe_ranking": rfe.ranking_
    }).sort_values("rfe_ranking", ascending=True).reset_index(drop=True)

    return df


def run_stability_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates selection stability across repeated stratified cross-validation folds.
    Computes empirical selection frequency for each candidate gene.
    """
    rng = np.random.RandomState(random_state)
    selection_counts = {g: 0 for g in feature_names}
    total_runs = 0

    for rep in range(n_repeats):
        seed = rng.randint(0, 100000)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, _ in skf.split(X, y):
            total_runs += 1
            X_tr, y_tr = X[train_idx], y[train_idx]
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.7, random_state=seed)
            clf.fit(X_tr, y_tr)
            selected_genes = [g for g, c in zip(feature_names, clf.coef_[0]) if c != 0]
            for g in selected_genes:
                selection_counts[g] += 1

    df = pd.DataFrame({
        "gene_symbol": feature_names,
        "selection_frequency": [selection_counts[g] / total_runs for g in feature_names]
    }).sort_values("selection_frequency", ascending=False).reset_index(drop=True)

    df["stability_rank"] = df["selection_frequency"].rank(ascending=False, method="min").astype(int)
    return df


def generate_biomarker_shortlist(
    disc_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate_genes: List[str],
    disc_dea_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Synthesizes multi-method feature selection outputs into a finalized,
    ranked biomarker candidate shortlist table.
    """
    X_disc = disc_df[candidate_genes].values
    y_disc = disc_df["y"].values

    scaler = StandardScaler()
    X_disc_std = scaler.fit_transform(X_disc)

    # 1. Feature selection passes
    lasso_df = run_lasso_feature_selection(X_disc_std, y_disc, candidate_genes, c_param=0.5, random_state=random_state)
    rf_df = run_random_forest_importance(X_disc_std, y_disc, candidate_genes, n_estimators=100, random_state=random_state)
    rfe_df = run_rfe_selection(X_disc_std, y_disc, candidate_genes, n_features_to_select=8, random_state=random_state)
    stability_df = run_stability_feature_selection(X_disc_std, y_disc, candidate_genes, n_splits=5, n_repeats=20, random_state=random_state)

    # Merge rankings
    merged = pd.DataFrame({"gene_symbol": candidate_genes})
    merged = merged.merge(lasso_df, on="gene_symbol")
    merged = merged.merge(rf_df, on="gene_symbol")
    merged = merged.merge(rfe_df, on="gene_symbol")
    merged = merged.merge(stability_df, on="gene_symbol")

    # Add discovery DEA stats
    dea_sub = disc_dea_df[["gene_symbol", "log2FC", "adj_p_val", "regulation"]].copy()
    dea_sub.columns = ["gene_symbol", "discovery_log2FC", "discovery_fdr", "regulation"]
    merged = merged.merge(dea_sub, on="gene_symbol", how="left")

    # Compute validation direction & effect size
    X_val = val_df[candidate_genes].values
    y_val = val_df["y"].values
    t2d_val = X_val[y_val == 1]
    ctrl_val = X_val[y_val == 0]
    val_log2fc = np.mean(t2d_val, axis=0) - np.mean(ctrl_val, axis=0)

    val_effect_df = pd.DataFrame({
        "gene_symbol": candidate_genes,
        "validation_log2FC": val_log2fc
    })
    merged = merged.merge(val_effect_df, on="gene_symbol", how="left")

    # Composite priority score: based on stability frequency, RF importance, and discovery significance
    # rank aggregation: lower composite rank is better
    merged["composite_rank"] = (
        merged["stability_rank"] * 0.35 +
        merged["rf_rank"] * 0.35 +
        merged["rfe_ranking"] * 0.15 +
        merged["lasso_rank"] * 0.15
    )
    merged = merged.sort_values("composite_rank", ascending=True).reset_index(drop=True)
    merged["priority_rank"] = np.arange(1, len(merged) + 1)

    # Panel assignments:
    # Top 5 = Core Panel
    # Top 8 = Extended Panel
    # Top 10 = Full Shortlist
    merged["panel_tier"] = "Exploratory"
    merged.loc[merged["priority_rank"] <= 10, "panel_tier"] = "Full-10"
    merged.loc[merged["priority_rank"] <= 8, "panel_tier"] = "Extended-8"
    merged.loc[merged["priority_rank"] <= 5, "panel_tier"] = "Core-5"
    # Compute direction concordance
    merged["is_direction_concordant"] = (np.sign(merged["discovery_log2FC"]) == np.sign(merged["validation_log2FC"]))

    # Functional annotation
    merged["functional_annotation"] = merged["gene_symbol"].map(
        lambda g: BIOLOGICAL_ANNOTATIONS.get(g, "Dysregulated metabolic/transcriptomic feature")
    )

    return merged
