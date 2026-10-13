"""
Stage 2: Clinical & Phenotypic Metadata Harmonization
Harmonizes sample characteristics, clinical arms, and disease stages across
GSE26168, GSE15932, and GSE164416 into a unified schema and validates alignment.
"""

from pathlib import Path
import gzip
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)
from preprocessing.common import log_step


def _parse_series_matrix_header(matrix_path: Path) -> Tuple[List[str], Dict[str, List[List[str]]]]:
    """
    Extracts sample accessions and metadata attributes from series matrix file header.
    """
    header_data: Dict[str, List[List[str]]] = defaultdict(list)
    with gzip.open(matrix_path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_"):
                parts = [p.replace('"', "").strip() for p in line.strip().split("\t")]
                header_data[parts[0]].append(parts[1:])
            elif line.startswith("!series_matrix_table_begin"):
                break

    accessions = header_data.get("!Sample_geo_accession", [[]])[0]
    return accessions, header_data


def _extract_per_sample_characteristics(header_data: Dict[str, Any], n_samples: int) -> List[Dict[str, str]]:
    """
    Safely indexes per-sample characteristics across all metadata rows.
    """
    sample_chars: List[Dict[str, str]] = [{} for _ in range(n_samples)]
    for row in header_data.get("!Sample_characteristics_ch1", []):
        for i in range(min(n_samples, len(row))):
            val = row[i].strip()
            if ":" in val:
                k, v = val.split(":", 1)
                sample_chars[i][k.strip().lower()] = v.strip()
            elif val:
                sample_chars[i]["unstructured"] = val
    return sample_chars


def harmonize_gse26168_metadata(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Harmonizes GSE26168 clinical metadata:
    - Normalizes 3 clinical tiers: Control, Impaired Fasting Glucose, Type 2 Diabetes
    - Normalizes tissue and gender
    """
    folder = config.resolve_raw_cohort_folder("GSE26168")
    cohort_cfg = config.cohorts["GSE26168"]
    matrix_files = list(folder.glob(cohort_cfg.series_matrix_pattern or "*series_matrix*.txt.gz"))
    if not matrix_files:
        matrix_files = list(folder.glob("*series_matrix*.txt.gz"))

    accessions, header_data = _parse_series_matrix_header(matrix_files[0])
    titles = header_data.get("!Sample_title", [[]])[0]
    sample_chars = _extract_per_sample_characteristics(header_data, len(accessions))

    records = []
    for i, gsm in enumerate(accessions):
        title = titles[i] if i < len(titles) else ""
        chars = sample_chars[i]
        raw_pt = chars.get("patient type", title)
        gender = chars.get("gender", "male")

        raw_pt_lower = raw_pt.lower()
        if "impaired fasting glucose" in raw_pt_lower:
            clinical_group = "Impaired Fasting Glucose"
            diabetes_status = "Pre-Diabetes"
        elif "type 2 diabetes" in raw_pt_lower or "t2d" in raw_pt_lower:
            clinical_group = "Type 2 Diabetes"
            diabetes_status = "Type 2 Diabetes"
        elif "control" in raw_pt_lower:
            clinical_group = "Control"
            diabetes_status = "Normoglycemic"
        else:
            clinical_group = "Unknown"
            diabetes_status = "Unknown"

        records.append({
            "sample_id": gsm,
            "cohort": "GSE26168",
            "platform_type": f"{cohort_cfg.modality} (Illumina HumanRef-8)",
            "platform_id": cohort_cfg.platform_id,
            "tissue_source": cohort_cfg.tissue_source,
            "donor_identifier": title,
            "clinical_group": clinical_group,
            "diabetes_status": diabetes_status,
            "cancer_status": "Cancer-Free",
            "sex": gender.capitalize(),
            "age_years": None,
            "fbs_mg_dl": None,
            "raw_title": title,
            "raw_characteristics": raw_pt
        })

    df = pd.DataFrame(records)
    log_step("STAGE-2: GSE26168", f"Harmonized metadata for {len(df)} samples: {dict(df['clinical_group'].value_counts())}")
    return df


def harmonize_gse15932_metadata(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Harmonizes GSE15932 clinical metadata:
    - Normalizes 4 clinical arms:
      1. Healthy Control
      2. Type 2 Diabetes
      3. Pancreatic Cancer (without Diabetes)
      4. Pancreatic Cancer with Diabetes
    - Parses numeric age, sex, fasting blood sugar (FBS), and diabetes duration
    """
    folder = config.resolve_raw_cohort_folder("GSE15932")
    cohort_cfg = config.cohorts["GSE15932"]
    matrix_files = list(folder.glob(cohort_cfg.series_matrix_pattern or "*series_matrix*.txt.gz"))
    accessions, header_data = _parse_series_matrix_header(matrix_files[0])
    titles = header_data.get("!Sample_title", [[]])[0]
    sample_chars = _extract_per_sample_characteristics(header_data, len(accessions))

    records = []
    for i, gsm in enumerate(accessions):
        title = titles[i] if i < len(titles) else ""
        chars = sample_chars[i]
        pt_id = chars.get("patient identifier", "")
        sex = chars.get("sex", "")
        age_str = chars.get("age of patient", "")
        fbs_str = chars.get("fbs", "")
        diag_str = chars.get("diagnosis", "")

        age_match = re.search(r"(\d+)", age_str)
        age_val = int(age_match.group(1)) if age_match else None

        fbs_match = re.search(r"(\d+(?:\.\d+)?)", fbs_str)
        fbs_val = float(fbs_match.group(1)) if fbs_match else None

        title_lower = title.lower()
        if "pancreatic cancer and diabetes" in title_lower:
            clinical_group = "Pancreatic Cancer with Diabetes"
            diabetes_status = "Type 2 Diabetes"
            cancer_status = "Pancreatic Adenocarcinoma"
        elif "patient with diabetes mellitus" in title_lower:
            clinical_group = "Type 2 Diabetes"
            diabetes_status = "Type 2 Diabetes"
            cancer_status = "Cancer-Free"
        elif "without diabetes" in title_lower:
            clinical_group = "Pancreatic Cancer (No Diabetes)"
            diabetes_status = "Normoglycemic"
            cancer_status = "Pancreatic Adenocarcinoma"
        elif "healthy control" in title_lower or not diag_str.strip():
            clinical_group = "Control"
            diabetes_status = "Normoglycemic"
            cancer_status = "Cancer-Free"
        else:
            clinical_group = "Other"
            diabetes_status = "Unknown"
            cancer_status = "Unknown"

        records.append({
            "sample_id": gsm,
            "cohort": "GSE15932",
            "platform_type": f"{cohort_cfg.modality} (Affymetrix HG-U133_Plus_2)",
            "platform_id": cohort_cfg.platform_id,
            "tissue_source": cohort_cfg.tissue_source,
            "donor_identifier": pt_id or title,
            "clinical_group": clinical_group,
            "diabetes_status": diabetes_status,
            "cancer_status": cancer_status,
            "sex": sex.capitalize() if sex else "Unknown",
            "age_years": age_val,
            "fbs_mg_dl": fbs_val,
            "raw_title": title,
            "raw_characteristics": f"diag: {diag_str} | fbs: {fbs_str}"
        })

    df = pd.DataFrame(records)
    log_step("STAGE-2: GSE15932", f"Harmonized metadata for {len(df)} samples: {dict(df['clinical_group'].value_counts())}")
    return df


def harmonize_gse164416_metadata(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Harmonizes GSE164416 clinical metadata:
    - Extracts donor IDs (DP002, DP003...) mapped from 133 HTSeq sample files
    """
    folder = config.resolve_raw_cohort_folder("GSE164416")
    cohort_cfg = config.cohorts["GSE164416"]
    count_files = sorted(folder.glob("*_htseq_counts.txt.gz"))

    records = []
    for filepath in count_files:
        match = re.search(r"(GSM\d+)_([A-Za-z0-9]+)_htseq_counts", filepath.name)
        if match:
            gsm_id, donor_id = match.group(1), match.group(2)
        else:
            gsm_id = filepath.name.split("_")[0]
            donor_id = "Unknown"

        records.append({
            "sample_id": gsm_id,
            "cohort": "GSE164416",
            "platform_type": f"{cohort_cfg.modality} (HTSeq counts)",
            "platform_id": cohort_cfg.platform_id,
            "tissue_source": cohort_cfg.tissue_source,
            "donor_identifier": donor_id,
            "clinical_group": "Islet Donor",
            "diabetes_status": "Donor Tissue",
            "cancer_status": "Non-Neoplastic Donor",
            "sex": "Unspecified",
            "age_years": None,
            "fbs_mg_dl": None,
            "raw_title": filepath.name,
            "raw_characteristics": f"Donor: {donor_id}"
        })

    df = pd.DataFrame(records)
    log_step("STAGE-2: GSE164416", f"Harmonized metadata for {len(df)} samples across islet donors")
    return df


def build_unified_clinical_metadata(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> pd.DataFrame:
    """
    Combines all cohort metadata tables into a single unified clinical schema.
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-2", "Building Unified Clinical Metadata Matrix...")

    df_26168 = harmonize_gse26168_metadata(config)
    df_15932 = harmonize_gse15932_metadata(config)
    df_164416 = harmonize_gse164416_metadata(config)

    unified_df = pd.concat([df_26168, df_15932, df_164416], ignore_index=True)

    output_path = dest / config.metadata.unified_metadata_filename
    unified_df.to_csv(output_path, index=False)

    # Mirror to alternate name if requested
    alt_output = dest / "clinical_metadata_unified.csv"
    if not alt_output.exists():
        unified_df.to_csv(alt_output, index=False)

    log_step("STAGE-2: SAVE", f"Master clinical metadata table saved to {output_path.name} ({len(unified_df)} total samples)")
    return unified_df


def validate_matrix_and_metadata_alignment(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> bool:
    """
    Ensures 100% sample identity alignment between the clinical metadata table
    and the columns of all Stage 1 expression matrices.
    """
    dest = config.paths.data_processed
    meta_path = dest / config.metadata.unified_metadata_filename
    if not meta_path.exists():
        meta_path = dest / "clinical_metadata_unified.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Neither {config.metadata.unified_metadata_filename} nor clinical_metadata_unified.csv exists.")

    meta_df = pd.read_csv(meta_path)
    meta_samples = set(meta_df["sample_id"])

    log_step("ALIGNMENT-CHECK", f"Checking alignment for {len(meta_samples)} metadata samples...")

    matrix_checks = [
        ("GSE26168", dest / config.cohorts["GSE26168"].raw_output_filename),
        ("GSE15932", dest / config.cohorts["GSE15932"].raw_output_filename),
        ("GSE164416", dest / config.cohorts["GSE164416"].raw_output_filename),
    ]

    all_matched = True
    for cohort_name, matrix_path in matrix_checks:
        if not matrix_path.exists():
            log_step("ALIGNMENT-CHECK", f"[FAIL] Missing matrix file: {matrix_path.name}")
            all_matched = False
            continue

        df_head = pd.read_csv(matrix_path, compression="gzip", nrows=1, index_col=0)
        matrix_cols = set(df_head.columns)

        cohort_meta_samples = set(meta_df[meta_df["cohort"] == cohort_name]["sample_id"])
        diff_missing_in_meta = matrix_cols - cohort_meta_samples
        diff_missing_in_matrix = cohort_meta_samples - matrix_cols

        if not diff_missing_in_meta and not diff_missing_in_matrix:
            log_step("ALIGNMENT-CHECK", f"[PASS] {cohort_name}: 100% sample match ({len(matrix_cols)} samples)")
        else:
            log_step("ALIGNMENT-CHECK", f"[MISMATCH] {cohort_name}: in matrix but not meta: {len(diff_missing_in_meta)}, in meta but not matrix: {len(diff_missing_in_matrix)}")
            all_matched = False

    return all_matched
