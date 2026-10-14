"""
Discovery Cohort Differential Expression Analysis Orchestrator
Conducts:
1. Primary statistical DEA on GSE26168 (T2D vs Control) using Limma Empirical Bayes
2. Head-to-head comparative analysis of low-expression filtering approaches:
   - Approach 1: Heuristic Background Floor (<= 80% background floor)
   - Approach 2: Distribution-Aware Non-Specific Variance Floor (IQR)
3. Monotonic early-dysregulation trajectory evaluation across Control -> IFG -> T2D
4. Delivery of publication-grade figures and structured CSV tables
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import scipy.stats as stats

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)
from preprocessing.normalization import preprocess_microarray_cohort
from .ebayes import run_ebayes_dea
from .visualizations import (
    plot_volcano,
    plot_ma,
    plot_deg_heatmap,
    plot_pvalue_distribution,
    plot_comparative_deg_yield
)


def log_dea(stage: str, message: str) -> None:
    print(f"[DEA-{stage}] {message}")


# Biological functional annotation dictionary for top prioritized candidate biomarkers
CANDIDATE_GENE_ANNOTATIONS = {
    "IRS1": "Insulin Receptor Substrate 1; central transducer of insulin/IGF-1 signaling; serine phosphorylation/downregulation drives peripheral insulin resistance",
    "INSR": "Insulin Receptor; transmembrane tyrosine kinase receptor mediating systemic insulin sensitivity and glycemic homeostasis",
    "FOXO1A": "Forkhead Box O1; pioneer transcription factor governing hepatic gluconeogenesis and beta-cell oxidative stress survival",
    "ANGPTL2": "Angiopoietin-Like Protein 2; pro-inflammatory adipokine inducing chronic vascular inflammation and peripheral insulin resistance",
    "IFNA7": "Interferon Alpha 7; innate type I interferon cytokine orchestrating antiviral and systemic sterile inflammation",
    "LRP5": "LDL Receptor-Related Protein 5; Wnt coreceptor critical for pancreatic beta-cell proliferation and glucose-stimulated insulin secretion",
    "PAQR9": "Progestin & AdipoQ Receptor Family Member 9; putative membrane adiponectin receptor family mediator in lipid/glucose metabolism",
    "DPT": "Dermatopontin; extracellular matrix protein modulating TGF-beta signaling and tissue remodeling in metabolic stress",
    "POPDC3": "Popeye Domain Containing 3; cAMP-binding membrane effector maintaining cardiac and cellular membrane integrity",
    "TSP50": "Testis-Specific Protease 50; oncofetal threonine protease implicated in NF-kB transcriptional activation and cell proliferation",
    "KISS1": "Kisspeptin-1; GPCR ligand regulating energy expenditure, puberty onset, and hypothalamic metabolic feedback",
    "MALAT1": "Metastasis-Associated Lung Adenocarcinoma Transcript 1; lncRNA regulator of glucose-induced endothelial and vascular inflammation",
    "DYRK3": "Dual-Specificity Tyrosine Phosphorylation Regulated Kinase 3; controls stress granule dynamics and mTORC1 reactivation",
    "MBNL3": "Muscleblind-Like Splicing Regulator 3; RNA-binding factor involved in developmental reprogramming and insulin alternative splicing",
    "SYNPO": "Synaptopodin; actin-associated protein linked to podocyte and cellular architectural stability under metabolic strain",
    "GABRA5": "Gamma-Aminobutyric Acid Type A Receptor Subunit Alpha5; neuro-endocrine and peripheral immune modulatory channel",
    "CES4": "Carboxylesterase 4; hydrolytic ester metabolism mediator implicated in lipid detoxification",
    "FUT11": "Fucosyltransferase 11; alpha-(1,3)-fucosyltransferase governing cell surface glycan remodeling and immune trafficking",
    "C6orf85": "Chromosome 6 Open Reading Frame 85 / FAM213A (PRDX9); redox regulatory protein mediating cellular antioxidant defense",
    "OR4D2": "Olfactory Receptor Family 4 Subfamily D Member 2; ectopic metabolic olfactory receptor in peripheral leukocytes",
    "OR8B12": "Olfactory Receptor Family 8 Subfamily B Member 12; chemosensory GPCR expressed in circulating immune cells",
    "AKT3": "RAC-gamma Serine/Threonine-Protein Kinase; downstream effector of PI3K/IRS1 pathway mediating glucose uptake and survival",
    "APOL4": "Apolipoprotein L4; lipid-binding protein implicated in intracellular lipid transport and innate immunity"
}


def load_gse26168_data_and_design(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """
    Loads raw expression matrix and experimental design for GSE26168.
    """
    dest = config.paths.data_processed

    # Design
    design_path = dest / "GSE26168_dea_design.csv"
    if not design_path.exists():
        from preprocessing.exports import export_differential_expression_designs
        from preprocessing.metadata import build_unified_clinical_metadata
        meta_df = build_unified_clinical_metadata(config)
        export_differential_expression_designs(meta_df, config)

    design_df = pd.read_csv(design_path)

    # Raw expression matrix
    raw_path = dest / config.cohorts["GSE26168"].raw_output_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw matrix {raw_path.name} not found in {dest}")

    log_dea("LOAD", f"Loading raw GSE26168 matrix from {raw_path.name}...")
    df_raw = pd.read_csv(raw_path, compression="gzip", index_col=0)

    # Samples
    case_samples = design_df[(design_df["condition"] == "T2D") & (design_df["is_primary_dea"])]["sample_id"].tolist()
    control_samples = design_df[(design_df["condition"] == "Control") & (design_df["is_primary_dea"])]["sample_id"].tolist()
    ifg_samples = design_df[design_df["condition"] == "IFG"]["sample_id"].tolist()

    log_dea("LOAD", f"Loaded design: {len(case_samples)} T2D Cases, {len(control_samples)} Controls, {len(ifg_samples)} IFG donors.")
    return df_raw, design_df, case_samples, control_samples, ifg_samples


def evaluate_early_trajectory(
    df_expr: pd.DataFrame,
    top_genes: List[str],
    design_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes mean expression and monotonic trend across the three clinical stages:
    Control (n=8) -> IFG (n=7) -> T2D (n=9).
    """
    stage_samples = {
        "Control": design_df[design_df["condition"] == "Control"]["sample_id"].tolist(),
        "IFG": design_df[design_df["condition"] == "IFG"]["sample_id"].tolist(),
        "T2D": design_df[design_df["condition"] == "T2D"]["sample_id"].tolist()
    }

    records = []
    for gene in top_genes:
        if gene not in df_expr.index:
            continue
        vals_ctrl = df_expr.loc[gene, stage_samples["Control"]].values
        vals_ifg = df_expr.loc[gene, stage_samples["IFG"]].values
        vals_t2d = df_expr.loc[gene, stage_samples["T2D"]].values

        all_vals = np.concatenate([vals_ctrl, vals_ifg, vals_t2d])
        stages = np.array([0] * len(vals_ctrl) + [1] * len(vals_ifg) + [2] * len(vals_t2d))

        spearman_rho, spearman_p = stats.spearmanr(stages, all_vals)
        f_stat, anova_p = stats.f_oneway(vals_ctrl, vals_ifg, vals_t2d)

        # Early shift direction: difference between IFG and Control
        early_delta = float(np.mean(vals_ifg) - np.mean(vals_ctrl))
        late_delta = float(np.mean(vals_t2d) - np.mean(vals_ifg))
        total_delta = float(np.mean(vals_t2d) - np.mean(vals_ctrl))

        records.append({
            "gene_symbol": gene,
            "mean_control": float(np.mean(vals_ctrl)),
            "mean_ifg": float(np.mean(vals_ifg)),
            "mean_t2d": float(np.mean(vals_t2d)),
            "std_control": float(np.std(vals_ctrl, ddof=1)),
            "std_ifg": float(np.std(vals_ifg, ddof=1)),
            "std_t2d": float(np.std(vals_t2d, ddof=1)),
            "delta_control_to_ifg": early_delta,
            "delta_ifg_to_t2d": late_delta,
            "total_delta_control_to_t2d": total_delta,
            "spearman_stage_rho": spearman_rho,
            "spearman_stage_pvalue": spearman_p,
            "anova_stage_pvalue": anova_p,
            "trajectory_pattern": "Monotonic Down" if spearman_rho < -0.5 else ("Monotonic Up" if spearman_rho > 0.5 else "Biphasic / Intermediate"),
            "functional_annotation": CANDIDATE_GENE_ANNOTATIONS.get(gene, "Transcriptomic biomarker candidate")
        })

    traj_df = pd.DataFrame(records).sort_values("spearman_stage_pvalue")
    return traj_df


def compare_filtering_approaches_dea(
    df_raw: pd.DataFrame,
    case_samples: List[str],
    control_samples: List[str],
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Executes a head-to-head DEA comparison of Approach 1 (Background Floor)
    vs Approach 2 (Variance Floor) on the discovery cohort.
    """
    log_dea("COMPARE", "Processing Approach 1: Heuristic Background Floor (<= 80% background floor)...")
    expr_appr1 = preprocess_microarray_cohort(
        "GSE26168",
        df_raw,
        config=config,
        filtering_approach="background",
        persist=False
    )
    dea_appr1 = run_ebayes_dea(expr_appr1, case_samples, control_samples)

    log_dea("COMPARE", "Processing Approach 2: Distribution-Aware Variance Floor (IQR max_mean)...")
    expr_appr2 = preprocess_microarray_cohort(
        "GSE26168",
        df_raw,
        config=config,
        filtering_approach="variance",
        persist=False
    )
    dea_appr2 = run_ebayes_dea(expr_appr2, case_samples, control_samples)

    sig1 = set(dea_appr1[dea_appr1["is_significant"]].index)
    sig2 = set(dea_appr2[dea_appr2["is_significant"]].index)
    shared_degs = sig1.intersection(sig2)

    summary_records = [
        {
            "filtering_approach": "Approach 1: Heuristic Background Floor",
            "filtering_strategy": "Empirical background floor (p20 threshold in >=20% samples)",
            "probe_aggregation": "mean",
            "features_tested": len(expr_appr1),
            "deg_count": len(sig1),
            "up_regulated": int((dea_appr1["regulation"] == "Up-regulated").sum()),
            "down_regulated": int((dea_appr1["regulation"] == "Down-regulated").sum()),
            "nominal_deg_count": int(dea_appr1["is_nominal"].sum()),
            "min_padj": float(dea_appr1["adj_p_val"].min()),
            "concordant_shared_with_other": len(shared_degs),
            "concordance_rate_pct": round(len(shared_degs) / len(sig1) * 100.0, 2) if len(sig1) > 0 else 0.0
        },
        {
            "filtering_approach": "Approach 2: Distribution-Aware Variance Floor",
            "filtering_strategy": "Non-specific variance pruning (bottom 20% IQR) + max_mean probe collapse",
            "probe_aggregation": "max_mean",
            "features_tested": len(expr_appr2),
            "deg_count": len(sig2),
            "up_regulated": int((dea_appr2["regulation"] == "Up-regulated").sum()),
            "down_regulated": int((dea_appr2["regulation"] == "Down-regulated").sum()),
            "nominal_deg_count": int(dea_appr2["is_nominal"].sum()),
            "min_padj": float(dea_appr2["adj_p_val"].min()),
            "concordant_shared_with_other": len(shared_degs),
            "concordance_rate_pct": round(len(shared_degs) / len(sig2) * 100.0, 2) if len(sig2) > 0 else 0.0
        }
    ]
    summary_df = pd.DataFrame(summary_records)
    log_dea("COMPARE", f"Comparative evaluation complete. Shared core DEGs: {len(shared_degs)}")
    return dea_appr1, dea_appr2, summary_df, len(shared_degs)


def run_discovery_dea_pipeline(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Dict[str, Any]:
    """
    Executes the complete differential expression analysis pipeline on the discovery dataset:
    1. Loads raw data and clinical designs.
    2. Runs comparative low-expression filtering and DEA under Approach 1 & Approach 2.
    3. Annotates significant candidate biomarkers with functional biological insights.
    4. Evaluates glycemic progression trajectory across Control -> IFG -> T2D.
    5. Saves full tables to results/tables/ and figures to results/figures/.
    """
    root = config.paths.project_root
    res_tables = root / "results" / "tables"
    res_figures = root / "results" / "figures"
    res_tables.mkdir(parents=True, exist_ok=True)
    res_figures.mkdir(parents=True, exist_ok=True)

    log_dea("MAIN", "==========================================================")
    log_dea("MAIN", "STARTING DIFFERENTIAL EXPRESSION ANALYSIS ON GSE26168")
    log_dea("MAIN", "==========================================================")

    # 1. Load Data
    df_raw, design_df, case_samples, control_samples, ifg_samples = load_gse26168_data_and_design(config)

    # 2. Comparative Analysis Across Both Filtering Approaches
    dea_appr1, dea_appr2, summary_df, n_overlap = compare_filtering_approaches_dea(
        df_raw, case_samples, control_samples, config
    )

    # Primary analysis is selected as Approach 1 (default config) or Approach 2
    primary_dea = dea_appr1
    primary_expr = preprocess_microarray_cohort("GSE26168", df_raw, config=config, filtering_approach="background", persist=False)

    # Significant DEGs
    sig_degs_df = primary_dea[primary_dea["is_significant"]].copy()
    sig_degs_df["functional_annotation"] = sig_degs_df["gene_symbol"].map(lambda g: CANDIDATE_GENE_ANNOTATIONS.get(g, "Dysregulated metabolic/transcriptomic feature"))

    # Trajectory across Control -> IFG -> T2D
    top_gene_list = sig_degs_df["gene_symbol"].tolist()
    traj_df = evaluate_early_trajectory(primary_expr, top_gene_list, design_df)

    # 3. Export Tabular Deliverables
    path_all = res_tables / "dea_discovery_gse26168_all_genes.csv"
    path_sig = res_tables / "dea_discovery_gse26168_significant_degs.csv"
    path_appr2_all = res_tables / "dea_discovery_gse26168_approach2_all_genes.csv"
    path_appr2_sig = res_tables / "dea_discovery_gse26168_approach2_significant_degs.csv"
    path_compare = res_tables / "dea_filtering_comparison_summary.csv"
    path_traj = res_tables / "dea_early_progression_trajectory_gse26168.csv"

    primary_dea.to_csv(path_all, index=False)
    sig_degs_df.to_csv(path_sig, index=False)
    dea_appr2.to_csv(path_appr2_all, index=False)
    dea_appr2[dea_appr2["is_significant"]].to_csv(path_appr2_sig, index=False)
    summary_df.to_csv(path_compare, index=False)
    traj_df.to_csv(path_traj, index=False)

    # Also mirror key tables to dataset/processed for notebook convenience
    mirror_dest = config.paths.dataset_processed
    if mirror_dest.exists():
        primary_dea.to_csv(mirror_dest / "GSE26168_dea_results_all.csv", index=False)
        sig_degs_df.to_csv(mirror_dest / "GSE26168_dea_results_sig.csv", index=False)

    log_dea("MAIN", f"Saved full DEA results to: {path_all.name}")
    log_dea("MAIN", f"Saved {len(sig_degs_df)} significant DEGs to: {path_sig.name}")
    log_dea("MAIN", f"Saved filtering comparison to: {path_compare.name}")
    log_dea("MAIN", f"Saved progression trajectory to: {path_traj.name}")

    # 4. Generate Publication-Grade Visualizations
    fig_volcano = res_figures / "dea_volcano_gse26168.png"
    fig_ma = res_figures / "dea_ma_plot_gse26168.png"
    fig_heatmap = res_figures / "dea_top_genes_heatmap_gse26168.png"
    fig_pvals = res_figures / "dea_pvalue_diagnostics_gse26168.png"
    fig_compare = res_figures / "dea_filtering_comparison_gse26168.png"

    plot_volcano(primary_dea, fig_volcano)
    plot_ma(primary_dea, fig_ma)
    plot_deg_heatmap(primary_expr, top_gene_list, design_df, fig_heatmap)
    plot_pvalue_distribution(primary_dea, fig_pvals)
    plot_comparative_deg_yield(summary_df, n_overlap, fig_compare)

    log_dea("MAIN", "==========================================================")
    log_dea("MAIN", "DIFFERENTIAL EXPRESSION ANALYSIS SUCCESSFULLY COMPLETED")
    log_dea("MAIN", "==========================================================")

    return {
        "tables": {
            "all_genes": path_all,
            "significant_degs": path_sig,
            "approach2_all": path_appr2_all,
            "approach2_sig": path_appr2_sig,
            "filtering_comparison": path_compare,
            "early_trajectory": path_traj
        },
        "figures": {
            "volcano": fig_volcano,
            "ma_plot": fig_ma,
            "deg_heatmap": fig_heatmap,
            "pvalue_diagnostics": fig_pvals,
            "filtering_comparison": fig_compare
        },
        "metrics": {
            "total_tested": len(primary_dea),
            "sig_degs": len(sig_degs_df),
            "up_regulated": int((sig_degs_df["regulation"] == "Up-regulated").sum()),
            "down_regulated": int((sig_degs_df["regulation"] == "Down-regulated").sum()),
            "shared_core_genes": n_overlap
        }
    }
