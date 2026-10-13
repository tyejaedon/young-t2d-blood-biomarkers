"""
Preprocessing Package
Modular stages for genomic and transcriptomic data preprocessing:
- extraction: Stage 1 cohort expression matrix extraction
- metadata: Stage 2 clinical & phenotypic metadata harmonization
- normalization: Stage 3 microarray & RNA-Seq normalization logic (with dual filtering & comparative analysis)
- exports: Stage 4 analysis-ready and ML-ready matrix exports
- common: shared logging and utility routines
"""

from preprocessing.common import log_step

from preprocessing.extraction import (
    extract_gse26168_expression,
    extract_gse15932_expression,
    extract_gse164416_counts,
    run_stage_1_extraction,
)

from preprocessing.metadata import (
    _parse_series_matrix_header,
    _extract_per_sample_characteristics,
    harmonize_gse26168_metadata,
    harmonize_gse15932_metadata,
    harmonize_gse164416_metadata,
    build_unified_clinical_metadata,
    validate_matrix_and_metadata_alignment,
)

from preprocessing.normalization import (
    log2_transform_microarray,
    quantile_normalize,
    load_gse26168_probe_map,
    load_gse15932_probe_map,
    map_probes_to_genes,
    filter_microarray_background,
    filter_microarray_variance,
    filter_microarray_genes,
    preprocess_microarray_cohort,
    filter_low_counts,
    filter_rnaseq_heuristic,
    filter_rnaseq_distribution_aware,
    filter_rnaseq_counts,
    calculate_deseq2_size_factors,
    normalize_deseq2,
    calculate_cpm_log2,
    preprocess_rnaseq_cohort,
    generate_comparative_filtering_matrices,
    run_stage_3_normalization,
    validate_stage_3_outputs,
)

from preprocessing.exports import (
    build_ml_ready_dataset,
    export_dataframe_formats,
    export_cross_cohort_blood_matrices,
    export_differential_expression_designs,
    mirror_processed_assets_to_dataset,
    run_stage_4_export,
    validate_stage_4_outputs,
)

__all__ = [
    "log_step",
    # Stage 1
    "extract_gse26168_expression",
    "extract_gse15932_expression",
    "extract_gse164416_counts",
    "run_stage_1_extraction",
    # Stage 2
    "_parse_series_matrix_header",
    "_extract_per_sample_characteristics",
    "harmonize_gse26168_metadata",
    "harmonize_gse15932_metadata",
    "harmonize_gse164416_metadata",
    "build_unified_clinical_metadata",
    "validate_matrix_and_metadata_alignment",
    # Stage 3
    "log2_transform_microarray",
    "quantile_normalize",
    "load_gse26168_probe_map",
    "load_gse15932_probe_map",
    "map_probes_to_genes",
    "filter_microarray_background",
    "filter_microarray_variance",
    "filter_microarray_genes",
    "preprocess_microarray_cohort",
    "filter_low_counts",
    "filter_rnaseq_heuristic",
    "filter_rnaseq_distribution_aware",
    "filter_rnaseq_counts",
    "calculate_deseq2_size_factors",
    "normalize_deseq2",
    "calculate_cpm_log2",
    "preprocess_rnaseq_cohort",
    "generate_comparative_filtering_matrices",
    "run_stage_3_normalization",
    "validate_stage_3_outputs",
    # Stage 4
    "build_ml_ready_dataset",
    "export_dataframe_formats",
    "export_cross_cohort_blood_matrices",
    "export_differential_expression_designs",
    "mirror_processed_assets_to_dataset",
    "run_stage_4_export",
    "validate_stage_4_outputs",
]
