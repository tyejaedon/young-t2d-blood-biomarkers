"""
Common utilities, reproducibility settings, and data loading for EDA pipeline.
"""

from pathlib import Path
import random
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Workspace directories
ROOT = Path("C:/Bioinformatics Project")
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_TABLES = ROOT / "results" / "tables"

RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


def log_eda(stage: str, message: str) -> None:
    """Standardized logging for EDA pipeline."""
    print(f"[EDA-{stage}] {message}")


def load_unified_metadata() -> pd.DataFrame:
    """
    Loads unified clinical metadata from data/processed.
    Accepts clinical_metadata_unified.csv or unified_clinical_metadata.csv.
    """
    candidates = [
        DATA_PROCESSED / "clinical_metadata_unified.csv",
        DATA_PROCESSED / "unified_clinical_metadata.csv"
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Ensure alternate path exists for interoperability
            alt_path = DATA_PROCESSED / "clinical_metadata_unified.csv"
            if not alt_path.exists():
                df.to_csv(alt_path, index=False)
            log_eda("METADATA", f"Loaded {len(df)} metadata records from {path.name}")
            return df
    raise FileNotFoundError(f"Unified clinical metadata not found in {DATA_PROCESSED}")


def load_expression_matrices() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Loads raw and normalized expression matrices for all three cohorts,
    plus the harmonized cross-cohort blood expression matrix.
    """
    log_eda("LOAD", "Loading raw and normalized matrices across all cohorts...")

    raw_matrices: Dict[str, pd.DataFrame] = {}
    norm_matrices: Dict[str, pd.DataFrame] = {}

    # GSE26168
    raw_26168 = DATA_PROCESSED / "GSE26168_raw_expression.csv.gz"
    norm_26168 = DATA_PROCESSED / "GSE26168_normalized_expression.csv.gz"
    raw_matrices["GSE26168"] = pd.read_csv(raw_26168, compression="gzip", index_col=0)
    norm_matrices["GSE26168"] = pd.read_csv(norm_26168, compression="gzip", index_col=0)

    # GSE15932
    raw_15932 = DATA_PROCESSED / "GSE15932_raw_expression.csv.gz"
    norm_15932 = DATA_PROCESSED / "GSE15932_normalized_expression.csv.gz"
    raw_matrices["GSE15932"] = pd.read_csv(raw_15932, compression="gzip", index_col=0)
    norm_matrices["GSE15932"] = pd.read_csv(norm_15932, compression="gzip", index_col=0)

    # GSE164416
    raw_164416 = DATA_PROCESSED / "GSE164416_raw_counts.csv.gz"
    norm_164416 = DATA_PROCESSED / "GSE164416_normalized_counts.csv.gz"
    raw_matrices["GSE164416"] = pd.read_csv(raw_164416, compression="gzip", index_col=0)
    norm_matrices["GSE164416"] = pd.read_csv(norm_164416, compression="gzip", index_col=0)

    # Cross-cohort harmonized blood matrix (13,585 genes x 56 blood samples)
    cross_path = DATA_PROCESSED / "cross_cohort_blood_expression_harmonized.csv"
    if not cross_path.exists():
        cross_path = DATA_PROCESSED / "cross_cohort_blood_expression_harmonized.csv.gz"
        cross_df = pd.read_csv(cross_path, compression="gzip", index_col=0)
    else:
        cross_df = pd.read_csv(cross_path, index_col=0)

    log_eda("LOAD", "All matrices loaded successfully.")
    return raw_matrices, norm_matrices, cross_df
