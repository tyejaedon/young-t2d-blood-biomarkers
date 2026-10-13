"""
Bioinformatics Data Preprocessing Pipeline Orchestrator
Coordinates Stage 1 through Stage 4:
- Stage 1: Cohort-Specific Expression Matrix Extraction & Storage (preprocessing.extraction)
- Stage 2: Clinical & Phenotypic Metadata Harmonization (preprocessing.metadata)
- Stage 3: Normalization & Preprocessing Logic (preprocessing.normalization)
- Stage 4: Matrix Export & Downstream ML Preparation (preprocessing.exports)
- Common: Standard logging (preprocessing.common)

Maintains full backwards compatibility by re-exporting all sub-module symbols.
"""

from typing import Dict, Any
from pathlib import Path
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)

# Re-export all sub-module functions for backward compatibility
from preprocessing import (
    log_step,
    # Stage 1
    extract_gse26168_expression,
    extract_gse15932_expression,
    extract_gse164416_counts,
    run_stage_1_extraction,
    # Stage 2
    _parse_series_matrix_header,
    _extract_per_sample_characteristics,
    harmonize_gse26168_metadata,
    harmonize_gse15932_metadata,
    harmonize_gse164416_metadata,
    build_unified_clinical_metadata,
    validate_matrix_and_metadata_alignment,
    # Stage 3
    log2_transform_microarray,
    quantile_normalize,
    load_gse26168_probe_map,
    load_gse15932_probe_map,
    map_probes_to_genes,
    preprocess_microarray_cohort,
    filter_low_counts,
    calculate_deseq2_size_factors,
    normalize_deseq2,
    calculate_cpm_log2,
    preprocess_rnaseq_cohort,
    run_stage_3_normalization,
    validate_stage_3_outputs,
    # Stage 4
    build_ml_ready_dataset,
    export_dataframe_formats,
    export_cross_cohort_blood_matrices,
    export_differential_expression_designs,
    mirror_processed_assets_to_dataset,
    run_stage_4_export,
    validate_stage_4_outputs,
)


def run_pipeline(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> Dict[str, Any]:
    """
    End-to-end Pipeline Orchestrator executing:
    - Stage 1: Cohort expression extraction
    - Stage 2: Clinical metadata harmonization & alignment validation
    - Stage 3: Cohort-specific normalization & preprocessing logic
    - Stage 4: Matrix export & downstream ML preparation
    - Comprehensive verification across all stages
    """
    log_step("PIPELINE", "Initiating Full Preprocessing Pipeline (Stages 1-4) with PreprocessingConfig")
    config.save_json()

    # Stage 1: Verify or run extraction
    sample_raw_matrix = config.paths.data_processed / config.cohorts["GSE26168"].raw_output_filename
    raw_results = None
    if not sample_raw_matrix.exists():
        raw_results = run_stage_1_extraction(config)

    # Stage 2: Harmonize clinical metadata
    unified_meta = build_unified_clinical_metadata(config)
    alignment_ok = validate_matrix_and_metadata_alignment(config)
    if not alignment_ok:
        log_step("PIPELINE: WARNING", "Detected alignment discrepancy between metadata and raw expression matrices.")

    # Stage 3: Normalization & Preprocessing Logic
    normalized_matrices = run_stage_3_normalization(config, raw_matrices=raw_results)
    stage_3_ok = validate_stage_3_outputs(config)

    # Stage 4: Matrix Export & Downstream ML Preparation
    manifest_df = run_stage_4_export(config, normalized_matrices=normalized_matrices, metadata_df=unified_meta)
    stage_4_ok = validate_stage_4_outputs(config)

    if alignment_ok and stage_3_ok and stage_4_ok:
        log_step("PIPELINE", "SUCCESS: Pipeline Stages 1, 2, 3, and 4 completed and verified with 0 errors.")
    else:
        log_step("PIPELINE: WARNING", "Pipeline finished with validation warnings.")

    return {
        "metadata": unified_meta,
        "normalized_matrices": normalized_matrices,
        "export_manifest": manifest_df
    }


if __name__ == "__main__":
    run_pipeline()
