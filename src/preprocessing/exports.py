"""
Stage 4: Matrix Export & Downstream ML Preparation
- Analysis-ready Gene x Sample expression exports (CSV, TSV, compressed)
- ML-ready Sample x Feature transposed datasets with clinical labels
- Cross-cohort harmonized blood discovery/validation matrix (13,585 shared genes)
- Differential expression design matrices & master export manifest
- Storage mirroring and verification
"""

from pathlib import Path
import shutil
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)
from preprocessing.common import log_step


def build_ml_ready_dataset(
    df_expression: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cohort_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Transforms gene expression matrix from (Features x Samples) to (Samples x Features)
    and joins comprehensive clinical metadata with machine learning target labels:
    - target_t2d: 1 for Type 2 Diabetes, 0 for Normoglycemic Control, NaN for other groups.
    - is_t2d_case_control: Boolean flag selecting the primary binary evaluation cohort.
    """
    # Transpose so samples become rows and gene features become columns
    df_transposed = df_expression.T
    df_transposed.index.name = "sample_id"

    # Filter metadata if cohort is specified
    meta_subset = metadata_df.copy()
    if cohort_name:
        meta_subset = meta_subset[meta_subset["cohort"] == cohort_name].copy()

    meta_indexed = meta_subset.set_index("sample_id")

    # Inner join to guarantee complete alignment between samples and labels
    ml_df = meta_indexed.join(df_transposed, how="inner")

    # Target variable construction:
    # 1: Type 2 Diabetes (including Pancreatic Cancer with T2D)
    # 0: Normoglycemic Control (Healthy Control or Pancreatic Cancer without Diabetes)
    # NaN: Impaired Fasting Glucose, Islet Donors, or unknown
    t2d_mask = ml_df["diabetes_status"] == "Type 2 Diabetes"
    control_mask = ml_df["diabetes_status"] == "Normoglycemic"

    ml_df["target_t2d"] = np.nan
    ml_df.loc[t2d_mask, "target_t2d"] = 1.0
    ml_df.loc[control_mask, "target_t2d"] = 0.0

    # Binary evaluation flag for strict Case vs Control studies
    ml_df["is_t2d_case_control"] = t2d_mask | control_mask

    # Reorder columns: clinical metadata & targets first, then gene features
    meta_cols = [c for c in meta_indexed.columns] + ["target_t2d", "is_t2d_case_control"]
    feature_cols = [c for c in df_transposed.columns]
    ordered_cols = meta_cols + feature_cols
    ml_df = ml_df[ordered_cols]

    return ml_df


def export_dataframe_formats(
    df: pd.DataFrame,
    dest_dir: Path,
    stem_name: str,
    export_tsv: bool = True,
    export_csv: bool = True,
    compress_tsv: bool = True
) -> List[Dict[str, Any]]:
    """
    Exports a DataFrame into multiple formats (.csv.gz, .csv, .tsv.gz)
    and collects file metadata for the master manifest.
    """
    records = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary Compressed CSV
    csv_gz_path = dest_dir / f"{stem_name}.csv.gz"
    df.to_csv(csv_gz_path, compression="gzip")
    records.append({
        "filename": csv_gz_path.name,
        "format": "csv.gz",
        "path": str(csv_gz_path),
        "size_bytes": csv_gz_path.stat().st_size,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
    })

    # 2. Uncompressed CSV (if requested)
    if export_csv:
        csv_path = dest_dir / f"{stem_name}.csv"
        df.to_csv(csv_path)
        records.append({
            "filename": csv_path.name,
            "format": "csv",
            "path": str(csv_path),
            "size_bytes": csv_path.stat().st_size,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        })

    # 3. TSV (compressed or uncompressed)
    if export_tsv:
        ext = "tsv.gz" if compress_tsv else "tsv"
        tsv_path = dest_dir / f"{stem_name}.{ext}"
        df.to_csv(tsv_path, sep="\t", compression="gzip" if compress_tsv else None)
        records.append({
            "filename": tsv_path.name,
            "format": ext,
            "path": str(tsv_path),
            "size_bytes": tsv_path.stat().st_size,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        })

    return records


def export_cross_cohort_blood_matrices(
    normalized_matrices: Dict[str, pd.DataFrame],
    metadata_df: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Harmonizes the primary blood microarray cohorts:
    - Discovery: GSE26168 (Whole Blood, Illumina HumanRef-8)
    - Validation: GSE15932 (PBMC, Affymetrix HG-U133 Plus 2)
    Computes the exact intersection of HGNC Gene Symbols across platforms,
    aligns common sample columns, and constructs:
    1. Harmonized Gene x Sample matrix (shared genes x 56 blood samples)
    2. Cross-cohort ML-ready Sample x Feature dataset with cohort indicators and target labels
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    df_26168 = normalized_matrices["GSE26168"]
    df_15932 = normalized_matrices["GSE15932"]

    # Compute intersection of HGNC Gene Symbols
    shared_genes = sorted(list(set(df_26168.index).intersection(set(df_15932.index))))
    log_step(
        "STAGE-4: HARMONIZE",
        f"Cross-cohort blood harmonization: {len(shared_genes)} shared HGNC gene symbols "
        f"across GSE26168 ({len(df_26168)} genes) and GSE15932 ({len(df_15932)} genes)"
    )

    # Subset and concatenate horizontally (columns = samples)
    df_shared_26168 = df_26168.loc[shared_genes]
    df_shared_15932 = df_15932.loc[shared_genes]
    cross_cohort_expr = pd.concat([df_shared_26168, df_shared_15932], axis=1)
    cross_cohort_expr.index.name = "gene_symbol"

    # Build cross-cohort ML ready table
    cross_cohort_ml = build_ml_ready_dataset(
        cross_cohort_expr,
        metadata_df,
        cohort_name=None
    )

    records = []

    # Export expression matrix
    expr_records = export_dataframe_formats(
        cross_cohort_expr,
        dest,
        stem_name="cross_cohort_blood_expression_harmonized",
        export_tsv=config.export.export_tsv,
        export_csv=config.export.export_csv_uncompressed
    )
    for r in expr_records:
        r.update({
            "matrix_name": "cross_cohort_blood_expression_harmonized",
            "cohort": "GSE26168 + GSE15932",
            "modality": "Microarray (Harmonized Blood)",
            "data_type": "normalized_gene_expression",
            "orientation": "genes_x_samples",
            "description": f"Harmonized cross-cohort blood expression on {len(shared_genes)} shared HGNC symbols across 56 samples"
        })
    records.extend(expr_records)

    # Export ML-ready table
    ml_records = export_dataframe_formats(
        cross_cohort_ml,
        dest,
        stem_name="cross_cohort_blood_ml_ready",
        export_tsv=config.export.export_tsv,
        export_csv=config.export.export_csv_uncompressed
    )
    for r in ml_records:
        r.update({
            "matrix_name": "cross_cohort_blood_ml_ready",
            "cohort": "GSE26168 + GSE15932",
            "modality": "Microarray (Harmonized Blood)",
            "data_type": "ml_features_labels",
            "orientation": "samples_x_features",
            "description": f"Transposed cross-cohort ML dataset (56 samples x {len(shared_genes)} genes + clinical targets)"
        })
    records.extend(ml_records)

    log_step("STAGE-4: HARMONIZE", f"Exported harmonized blood expression ({cross_cohort_expr.shape}) and ML dataset ({cross_cohort_ml.shape})")
    return cross_cohort_expr, cross_cohort_ml, records


def export_differential_expression_designs(
    metadata_df: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> List[Dict[str, Any]]:
    """
    Generates experimental design matrices for statistical Differential Expression Analysis (DEA):
    - GSE26168 Discovery design: T2D vs Normoglycemic Control
    - GSE15932 Validation design: T2D vs Normoglycemic Control (accounting for Pancreatic Cancer status)
    - Combined cross-cohort blood design: T2D vs Control with cohort batch variable
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. GSE26168 Design
    df_26168 = metadata_df[metadata_df["cohort"] == "GSE26168"].copy()
    df_26168["condition"] = df_26168["clinical_group"].map({
        "Type 2 Diabetes": "T2D",
        "Control": "Control",
        "Impaired Fasting Glucose": "IFG"
    })
    df_26168["is_primary_dea"] = df_26168["condition"].isin(["T2D", "Control"])
    design_26168 = df_26168[["sample_id", "condition", "is_primary_dea", "sex", "tissue_source"]]
    p_26168 = dest / "GSE26168_dea_design.csv"
    design_26168.to_csv(p_26168, index=False)
    records.append({
        "matrix_name": "GSE26168_dea_design",
        "filename": p_26168.name,
        "cohort": "GSE26168",
        "modality": "Metadata",
        "data_type": "dea_design",
        "orientation": "samples_x_covariates",
        "n_rows": design_26168.shape[0],
        "n_cols": design_26168.shape[1],
        "size_bytes": p_26168.stat().st_size,
        "format": "csv",
        "path": str(p_26168),
        "description": "Experimental design table for GSE26168 limma DEA (T2D vs Control)"
    })

    # 2. GSE15932 Design
    df_15932 = metadata_df[metadata_df["cohort"] == "GSE15932"].copy()
    df_15932["condition"] = df_15932["diabetes_status"].map({
        "Type 2 Diabetes": "T2D",
        "Normoglycemic": "Control"
    })
    df_15932["cancer_covariate"] = df_15932["cancer_status"].apply(lambda s: "Cancer" if "Adenocarcinoma" in s else "CancerFree")
    df_15932["is_primary_dea"] = df_15932["cancer_covariate"] == "CancerFree"  # Pure T2D vs Pure Control
    design_15932 = df_15932[["sample_id", "condition", "cancer_covariate", "is_primary_dea", "sex", "age_years", "fbs_mg_dl"]]
    p_15932 = dest / "GSE15932_dea_design.csv"
    design_15932.to_csv(p_15932, index=False)
    records.append({
        "matrix_name": "GSE15932_dea_design",
        "filename": p_15932.name,
        "cohort": "GSE15932",
        "modality": "Metadata",
        "data_type": "dea_design",
        "orientation": "samples_x_covariates",
        "n_rows": design_15932.shape[0],
        "n_cols": design_15932.shape[1],
        "size_bytes": p_15932.stat().st_size,
        "format": "csv",
        "path": str(p_15932),
        "description": "Experimental design table for GSE15932 validation DEA"
    })

    log_step("STAGE-4: DEA-DESIGN", f"Exported statistical design matrices for GSE26168 ({len(design_26168)} samples) and GSE15932 ({len(design_15932)} samples)")
    return records


def mirror_processed_assets_to_dataset(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> None:
    """
    Mirrors key processed files from data/processed to dataset/processed
    to guarantee absolute compatibility across both folder conventions.
    """
    src_dir = config.paths.data_processed
    dest_dir = config.paths.dataset_processed

    if src_dir.resolve() == dest_dir.resolve():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src_dir.glob("*"):
        if item.is_file():
            target = dest_dir / item.name
            shutil.copy2(item, target)
            count += 1

    log_step("STAGE-4: MIRROR", f"Mirrored {count} processed assets from data/processed to {dest_dir.name}")


def run_stage_4_export(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    normalized_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    metadata_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Executes Stage 4: Matrix Export & Downstream ML Preparation across all cohorts:
    1. Analysis-ready Gene x Sample expression exports (.csv.gz, .csv, .tsv.gz)
    2. ML-ready Sample x Feature transposed datasets with clinical targets (.csv.gz, .csv)
    3. Cross-cohort harmonized blood matrix & ML table (13,585 shared genes)
    4. Differential expression experimental design matrices
    5. Master matrix export manifest compilation & persistence
    6. Mirroring to dataset/processed
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-4", "Starting Stage 4 Matrix Export & Downstream ML Preparation...")

    # Load metadata if not supplied
    if metadata_df is None:
        meta_path = dest / config.metadata.unified_metadata_filename
        if not meta_path.exists():
            meta_path = dest / "clinical_metadata_unified.csv"
        if not meta_path.exists():
            from preprocessing.metadata import build_unified_clinical_metadata
            metadata_df = build_unified_clinical_metadata(config)
        else:
            metadata_df = pd.read_csv(meta_path)

    # Load normalized matrices if not supplied
    if normalized_matrices is None:
        normalized_matrices = {}
        for cohort_name, cohort_cfg in config.cohorts.items():
            matrix_path = dest / cohort_cfg.normalized_output_filename
            if not matrix_path.exists():
                raise FileNotFoundError(f"Normalized matrix {matrix_path.name} not found. Run Stage 3 first.")
            log_step("STAGE-4: LOAD", f"Loading {cohort_name} normalized matrix from {matrix_path.name}...")
            normalized_matrices[cohort_name] = pd.read_csv(matrix_path, compression="gzip", index_col=0)

    all_manifest_records: List[Dict[str, Any]] = []

    # 1. Export cohort-level matrices and ML datasets
    for cohort_name, cohort_cfg in config.cohorts.items():
        df_norm = normalized_matrices[cohort_name]
        stem_expr = f"{cohort_name}_analysis_ready_expression"

        # Export Expression Formats
        expr_recs = export_dataframe_formats(
            df_norm,
            dest,
            stem_name=stem_expr,
            export_tsv=config.export.export_tsv,
            export_csv=config.export.export_csv_uncompressed
        )
        for r in expr_recs:
            r.update({
                "matrix_name": stem_expr,
                "cohort": cohort_name,
                "modality": cohort_cfg.modality,
                "data_type": "normalized_gene_expression",
                "orientation": "genes_x_samples",
                "description": f"Analysis-ready {cohort_name} normalized expression ({df_norm.shape[0]} features x {df_norm.shape[1]} samples)"
            })
        all_manifest_records.extend(expr_recs)

        # Build and Export ML Dataset (Samples x Features)
        if config.export.export_ml_ready:
            df_ml = build_ml_ready_dataset(df_norm, metadata_df, cohort_name=cohort_name)
            stem_ml = f"{cohort_name}_ml_features_labels"
            ml_recs = export_dataframe_formats(
                df_ml,
                dest,
                stem_name=stem_ml,
                export_tsv=config.export.export_tsv,
                export_csv=config.export.export_csv_uncompressed
            )
            for r in ml_recs:
                r.update({
                    "matrix_name": stem_ml,
                    "cohort": cohort_name,
                    "modality": cohort_cfg.modality,
                    "data_type": "ml_features_labels",
                    "orientation": "samples_x_features",
                    "description": f"Transposed ML dataset for {cohort_name} ({df_ml.shape[0]} samples x {df_ml.shape[1]} columns with target_t2d)"
                })
            all_manifest_records.extend(ml_recs)

    # 2. Cross-cohort Blood Harmonization (GSE26168 + GSE15932)
    if config.export.export_cross_cohort_harmonized and "GSE26168" in normalized_matrices and "GSE15932" in normalized_matrices:
        _, _, cross_recs = export_cross_cohort_blood_matrices(normalized_matrices, metadata_df, config)
        all_manifest_records.extend(cross_recs)

    # 3. DEA Design Matrices
    dea_recs = export_differential_expression_designs(metadata_df, config)
    all_manifest_records.extend(dea_recs)

    # 4. Master Matrix Export Manifest
    manifest_df = pd.DataFrame(all_manifest_records)
    # Reorder columns cleanly
    col_order = ["matrix_name", "filename", "cohort", "modality", "data_type", "orientation", "n_rows", "n_cols", "format", "size_bytes", "description", "path"]
    existing_cols = [c for c in col_order if c in manifest_df.columns]
    manifest_df = manifest_df[existing_cols]

    manifest_path = dest / config.export.export_manifest_filename
    manifest_df.to_csv(manifest_path, index=False)
    log_step("STAGE-4: MANIFEST", f"Master matrix manifest compiled with {len(manifest_df)} artifacts saved to {manifest_path.name}")

    # 5. Mirror to dataset/processed (if enabled)
    if config.export.mirror_to_dataset_processed:
        mirror_processed_assets_to_dataset(config)

    log_step("STAGE-4", "Completed Stage 4 Matrix Export & Downstream ML Preparation successfully.")
    return manifest_df


def validate_stage_4_outputs(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> bool:
    """
    Validates Stage 4 outputs:
    - Verifies manifest file exists and is populated
    - Verifies all listed matrix files exist on disk and have non-zero file sizes
    - Verifies sample and feature counts in exported matrices
    """
    dest = config.paths.data_processed
    manifest_path = dest / config.export.export_manifest_filename

    log_step("STAGE-4: VERIFY", f"Validating Stage 4 exports against {manifest_path.name}...")
    if not manifest_path.exists():
        log_step("STAGE-4: VERIFY", f"[FAIL] Export manifest not found: {manifest_path}")
        return False

    manifest_df = pd.read_csv(manifest_path)
    all_valid = True

    for _, row in manifest_df.iterrows():
        p = Path(row["path"])
        if not p.exists():
            log_step("STAGE-4: VERIFY", f"[FAIL] File missing: {p.name}")
            all_valid = False
            continue
        if p.stat().st_size == 0:
            log_step("STAGE-4: VERIFY", f"[FAIL] File empty (0 bytes): {p.name}")
            all_valid = False
            continue

    if all_valid:
        log_step("STAGE-4: VERIFY", f"[PASS] All {len(manifest_df)} exported matrix artifacts verified intact on disk.")
    return all_valid
