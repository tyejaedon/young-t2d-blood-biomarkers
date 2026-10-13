"""
Stage 1: Cohort-Specific Expression Matrix Extraction & Storage
Extracts raw series matrix expression tables and per-sample HTSeq count files
for GSE26168, GSE15932, and GSE164416.
"""

from pathlib import Path
import gzip
import io
import re
from typing import Dict, List, Optional
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)
from preprocessing.common import log_step


def extract_gse26168_expression(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Extracts expression data for GSE26168 from its series matrix file.
    Prefers the primary platform GPL6883.
    """
    cohort_cfg = config.cohorts["GSE26168"]
    folder = config.resolve_raw_cohort_folder("GSE26168")
    
    matrix_files = list(folder.glob(cohort_cfg.series_matrix_pattern or "*series_matrix*.txt.gz"))
    if not matrix_files:
        matrix_files = list(folder.glob("*series_matrix*.txt.gz"))

    if not matrix_files:
        raise FileNotFoundError(f"No series matrix file found in {folder}")

    target_matrix = matrix_files[0]
    log_step("STAGE-1: GSE26168", f"Reading expression table from {target_matrix.name}...")

    table_lines = []
    capture = False
    with gzip.open(target_matrix, "rt", errors="ignore") as f:
        for line in f:
            if "!series_matrix_table_begin" in line:
                capture = True
                continue
            if "!series_matrix_table_end" in line:
                break
            if capture:
                table_lines.append(line)

    if not table_lines:
        raise ValueError(f"No table content found in {target_matrix.name}")

    df = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    df.index.name = "probe_id"
    df.columns = [c.replace('"', "").strip() for c in df.columns]
    df.index = [str(idx).replace('"', "").strip() for idx in df.index]

    log_step("STAGE-1: GSE26168", f"Extracted expression matrix: {df.shape[0]} probes x {df.shape[1]} samples")
    return df


def extract_gse15932_expression(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Extracts expression data for GSE15932 from its series matrix file.
    """
    cohort_cfg = config.cohorts["GSE15932"]
    folder = config.resolve_raw_cohort_folder("GSE15932")
    matrix_files = list(folder.glob(cohort_cfg.series_matrix_pattern or "*series_matrix*.txt.gz"))

    if not matrix_files:
        raise FileNotFoundError(f"No series matrix file found in {folder}")

    target_matrix = matrix_files[0]
    log_step("STAGE-1: GSE15932", f"Reading expression table from {target_matrix.name}...")

    table_lines = []
    capture = False
    with gzip.open(target_matrix, "rt", errors="ignore") as f:
        for line in f:
            if "!series_matrix_table_begin" in line:
                capture = True
                continue
            if "!series_matrix_table_end" in line:
                break
            if capture:
                table_lines.append(line)

    if not table_lines:
        raise ValueError(f"No table content found in {target_matrix.name}")

    df = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
    df.index.name = "probe_id"
    df.columns = [c.replace('"', "").strip() for c in df.columns]
    df.index = [str(idx).replace('"', "").strip() for idx in df.index]

    log_step("STAGE-1: GSE15932", f"Extracted expression matrix: {df.shape[0]} probes x {df.shape[1]} samples")
    return df


def extract_gse164416_counts(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Aggregates 133 individual HTSeq count files into a unified Gene x Sample matrix.
    Filters out HTSeq metadata rows (__no_feature, __ambiguous, etc.).
    """
    folder = config.resolve_raw_cohort_folder("GSE164416")
    count_files = sorted(folder.glob("*_htseq_counts.txt.gz"))

    if not count_files:
        raise FileNotFoundError(f"No htseq count files found in {folder}")

    log_step("STAGE-1: GSE164416", f"Aggregating {len(count_files)} per-sample HTSeq count files...")

    sample_dict = {}
    for filepath in count_files:
        match = re.search(r"(GSM\d+)", filepath.name)
        if not match:
            continue
        gsm_id = match.group(1)

        df_sample = pd.read_csv(filepath, sep="\t", header=0, names=["ensembl_id", "counts"], compression="gzip")
        df_sample = df_sample[~df_sample["ensembl_id"].str.startswith("__")]
        sample_dict[gsm_id] = df_sample.set_index("ensembl_id")["counts"]

    merged_df = pd.DataFrame(sample_dict)
    merged_df.index.name = "ensembl_id"

    log_step("STAGE-1: GSE164416", f"Aggregated count matrix: {merged_df.shape[0]} genes x {merged_df.shape[1]} samples")
    return merged_df


def run_stage_1_extraction(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> Dict[str, pd.DataFrame]:
    """
    Executes Stage 1 across all 3 cohorts and saves raw extracted expression tables as compressed CSVs.
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-1", "Starting Stage 1 Expression Extraction Pipeline")
    results = {}

    # GSE26168
    df_gse26168 = extract_gse26168_expression(config)
    out_26168 = dest / config.cohorts["GSE26168"].raw_output_filename
    df_gse26168.to_csv(out_26168, compression="gzip")
    results["GSE26168"] = df_gse26168

    # GSE15932
    df_gse15932 = extract_gse15932_expression(config)
    out_15932 = dest / config.cohorts["GSE15932"].raw_output_filename
    df_gse15932.to_csv(out_15932, compression="gzip")
    results["GSE15932"] = df_gse15932

    # GSE164416
    df_gse164416 = extract_gse164416_counts(config)
    out_164416 = dest / config.cohorts["GSE164416"].raw_output_filename
    df_gse164416.to_csv(out_164416, compression="gzip")
    results["GSE164416"] = df_gse164416

    log_step("STAGE-1", f"All raw expression matrices saved to {dest}")
    return results
