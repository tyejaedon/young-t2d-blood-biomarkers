"""
Bioinformatics Data Preprocessing Pipeline
Stage 1: Cohort-Specific Expression Matrix Extraction & Storage
"""

from pathlib import Path
import gzip
import io
import re
from typing import Dict, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(r"C:/Bioinformatics Project")
DATASET_RAW = PROJECT_ROOT / "dataset" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def log_step(stage: str, message: str) -> None:
    """Standardized clean logging."""
    print(f"[{stage}] {message}")


# -----------------------------------------------------------------------------
# 1. GSE26168 Matrix Extractor (Illumina Microarray via Series Matrix)
# -----------------------------------------------------------------------------
def extract_gse26168_expression(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Extracts expression data for GSE26168 from its series matrix file.
    Prefers the primary platform GPL6883.
    GSE26168 is an Illumina microarray dataset, and the series matrix file contains the expression values for all samples.
    An expression is a table where rows are probes and columns are samples.
    A probe is a specific DNA sequence on the microarray that corresponds to a gene or transcript.
    """
    folder = raw_dir or (DATASET_RAW / "GSE26168_RAW")
    matrix_files = list(folder.glob("*GPL6883*series_matrix*.txt.gz"))
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
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    df.index = [str(idx).replace('"', '').strip() for idx in df.index]

    log_step("STAGE-1: GSE26168", f"Successfully extracted matrix: {df.shape[0]} probes x {df.shape[1]} samples")
    return df


# -----------------------------------------------------------------------------
# 2. GSE15932 Matrix Extractor (Affymetrix Microarray via Series Matrix)
# -----------------------------------------------------------------------------
def extract_gse15932_expression(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Extracts expression data for GSE15932 from its series matrix file.
    GSE15932 is an Affymetrix microarray dataset, and the series matrix file contains the expression values for all samples.
    An Affymetrix microarray is a type of DNA microarray that uses short oligonucleotide probes to measure gene expression.
    An oligonucleotide probe is a short sequence of nucleotides that can hybridize to a specific target sequence in the sample.
    For example , a probe might be designed to bind to the mRNA of a specific gene, allowing the microarray to measure how much of that gene is being expressed in the sample.
    """
    folder = raw_dir or (DATASET_RAW / "GSE15932_RAW")
    matrix_files = list(folder.glob("*series_matrix*.txt.gz"))

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
    df.columns = [c.replace('"', '').strip() for c in df.columns]
    df.index = [str(idx).replace('"', '').strip() for idx in df.index]

    log_step("STAGE-1: GSE15932", f"Successfully extracted matrix: {df.shape[0]} probes x {df.shape[1]} samples")
    return df


# -----------------------------------------------------------------------------
# 3. GSE164416 Count Matrix Extractor (RNA-Seq HTSeq per-sample counts)
# -----------------------------------------------------------------------------
def extract_gse164416_counts(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Aggregates 133 individual HTSeq count files into a unified Gene x Sample matrix.
    Filters out HTSeq metadata rows (__no_feature, __ambiguous, etc.).
    A HTSeq count file is a tab-delimited text file that contains the number of reads mapped to each gene for a single RNA-Seq sample.
    Each row corresponds to a gene, and the columns typically include the gene identifier and the count
    """
    folder = raw_dir or (DATASET_RAW / "GSE164416_RAW")
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

    log_step("STAGE-1: GSE164416", f"Successfully aggregated count matrix: {merged_df.shape[0]} genes x {merged_df.shape[1]} samples")
    return merged_df


# -----------------------------------------------------------------------------
# Stage 1 Orchestration & Artifact Persistence
# -----------------------------------------------------------------------------
def run_stage_1_extraction(output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Executes Stage 1 across all 3 cohorts and saves raw extracted expression tables as compressed CSVs.
    """
    dest = output_dir or DATA_PROCESSED
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-1", "Starting Stage 1 Expression Extraction Pipeline")
    results = {}

    # GSE26168
    df_gse26168 = extract_gse26168_expression()
    out_26168 = dest / "GSE26168_raw_expression.csv.gz"
    df_gse26168.to_csv(out_26168, compression="gzip")
    log_step("STAGE-1: SAVE", f"Saved GSE26168 to {out_26168.name}")
    results["GSE26168"] = df_gse26168

    # GSE15932
    df_gse15932 = extract_gse15932_expression()
    out_15932 = dest / "GSE15932_raw_expression.csv.gz"
    df_gse15932.to_csv(out_15932, compression="gzip")
    log_step("STAGE-1: SAVE", f"Saved GSE15932 to {out_15932.name}")
    results["GSE15932"] = df_gse15932

    # GSE164416
    df_gse164416 = extract_gse164416_counts()
    out_164416 = dest / "GSE164416_raw_counts.csv.gz"
    df_gse164416.to_csv(out_164416, compression="gzip")
    log_step("STAGE-1: SAVE", f"Saved GSE164416 to {out_164416.name}")
    results["GSE164416"] = df_gse164416

    log_step("STAGE-1", f"All raw expression matrices saved to {dest}")
    return results


if __name__ == "__main__":
    run_stage_1_extraction()
