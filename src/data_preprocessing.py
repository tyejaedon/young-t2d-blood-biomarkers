"""
Bioinformatics Data Preprocessing Pipeline
Stage 1: Cohort-Specific Expression Matrix Extraction & Storage
Stage 2: Clinical & Phenotypic Metadata Harmonization
Stage 3: Normalization & Preprocessing Logic
    - Microarray: Log2 transformation, quantile normalization, probe mapping to HGNC Gene Symbols
    - RNA-Seq: Low-count filtering, DESeq2 size-factor normalization & CPM log2 transformation
Config: Driven by PreprocessingConfig (preprocessing_config.py)
"""

from pathlib import Path
import gzip
import io
import re
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
import numpy as np
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    CohortSettings
)


def log_step(stage: str, message: str) -> None:
    """Standardized clean logging."""
    print(f"[{stage}] {message}")


# =============================================================================
# STAGE 1: COHORT-SPECIFIC EXPRESSION EXTRACTION
# =============================================================================

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


# =============================================================================
# STAGE 2: CLINICAL & PHENOTYPIC METADATA HARMONIZATION
# =============================================================================

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
        raise FileNotFoundError(f"{meta_path} does not exist.")

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


# =============================================================================
# STAGE 3: NORMALIZATION & PREPROCESSING LOGIC
# =============================================================================

# --- Microarray Normalization & Probe Mapping Logic ---

def log2_transform_microarray(df: pd.DataFrame, offset: float = 1.0) -> pd.DataFrame:
    """
    Applies log2 transformation to microarray intensity matrices.
    Safely clips negative values (e.g. from background subtraction) at 0
    before adding offset and taking np.log2.
    Detects if the matrix is already log-transformed (max < 30 and min >= 0).
    """
    max_val = df.max().max()
    min_val = df.min().min()

    # Heuristic for detecting prior log2 transformation:
    # Microarray raw intensities typically range up to several thousands (e.g. > 100),
    # while log2 transformed values are typically between 0 and 20.
    if max_val <= 30.0 and min_val >= 0.0:
        log_step("STAGE-3: LOG2", f"Data already appears log2 transformed (min={min_val:.2f}, max={max_val:.2f}). Skipping.")
        return df.copy()

    log_step("STAGE-3: LOG2", f"Applying log2(intensity + {offset}) [pre-transform range: {min_val:.2f} to {max_val:.2f}]")
    clipped = np.clip(df.values, a_min=0.0, a_max=None)
    transformed = np.log2(clipped + offset)
    return pd.DataFrame(transformed, index=df.index, columns=df.columns)


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Bolstad et al. (2003) quantile normalization across samples (columns).
    Equalizes empirical distributions across all arrays while preserving within-sample ranks.
    Uses rank averaging and linear interpolation for tied values.
    """
    log_step("STAGE-3: QNORM", f"Quantile normalizing matrix of shape {df.shape[0]} features x {df.shape[1]} samples")
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols <= 1:
        return df.copy()

    # Step 1: Sort each column independently
    sorted_values = np.sort(df.values, axis=0)

    # Step 2: Compute mean expression at each rank across samples
    target_distribution = np.mean(sorted_values, axis=1)

    # Step 3: Compute ranks with average tie-breaking
    ranks = df.rank(method="average", axis=0).to_numpy() - 1.0
    grid = np.arange(n_rows, dtype=float)

    # Step 4: Map rank percentiles back onto target distribution
    normalized = np.empty((n_rows, n_cols), dtype=float)
    for col_idx in range(n_cols):
        normalized[:, col_idx] = np.interp(ranks[:, col_idx], grid, target_distribution)

    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def load_gse26168_probe_map(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> Dict[str, str]:
    """
    Parses Illumina GPL6883 HumanRef-8 v3 BGX manifest to map Probe_Id -> HGNC Symbol.
    Checks uncompressed and compressed .bgx files.
    """
    folder = config.resolve_raw_cohort_folder("GSE26168")
    bgx_candidates = [
        folder / "uncompressed" / "GPL6883_HumanRef-8_V3_0_R0_11282963_A.bgx",
        folder / "GPL6883_HumanRef-8_V3_0_R0_11282963_A.bgx",
        folder / "GPL6883_HumanRef-8_V3_0_R0_11282963_A.bgx.gz",
    ]
    bgx_path = None
    for candidate in bgx_candidates:
        if candidate.exists():
            bgx_path = candidate
            break

    if not bgx_path:
        raise FileNotFoundError(f"GPL6883 BGX manifest not found in {folder}")

    log_step("STAGE-3: PROBE-MAP", f"Loading GSE26168 probe annotations from {bgx_path.name}...")
    opener = gzip.open if bgx_path.suffix == ".gz" else open

    probe_map = {}
    with opener(bgx_path, "rt", errors="ignore") as f:
        in_probes = False
        probe_idx, symbol_idx, ilmn_gene_idx = -1, -1, -1
        for line in f:
            line_str = line.strip()
            if line_str.startswith("[Probes]"):
                in_probes = True
                header = f.readline().strip().split("\t")
                probe_idx = header.index("Probe_Id")
                symbol_idx = header.index("Symbol") if "Symbol" in header else -1
                ilmn_gene_idx = header.index("ILMN_Gene") if "ILMN_Gene" in header else -1
                continue

            if in_probes:
                if line_str.startswith("["):
                    break
                parts = line_str.split("\t")
                if len(parts) > probe_idx:
                    pid = parts[probe_idx].strip()
                    sym = ""
                    if symbol_idx != -1 and len(parts) > symbol_idx:
                        sym = parts[symbol_idx].strip()
                    if not sym and ilmn_gene_idx != -1 and len(parts) > ilmn_gene_idx:
                        sym = parts[ilmn_gene_idx].strip()
                    if sym:
                        probe_map[pid] = sym

    log_step("STAGE-3: PROBE-MAP", f"GSE26168: Loaded mapping for {len(probe_map)} probes")
    return probe_map


def load_gse15932_probe_map(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> Dict[str, str]:
    """
    Parses Affymetrix GPL570 (HG-U133_Plus_2) annotation table to map ID -> HGNC Symbol.
    Extracts primary symbol when multiple symbols are delimited with '///'.
    """
    # Look for cached GPL570 in dataset/raw, data/raw, or cohort folder
    search_paths = [
        config.paths.dataset_raw / "GPL570.txt",
        config.paths.data_raw_alt / "GPL570.txt",
        config.resolve_raw_cohort_folder("GSE15932") / "GPL570.txt",
        config.paths.dataset_raw / "GPL570_RAW.txt",
    ]
    gpl_path = None
    for p in search_paths:
        if p.exists():
            gpl_path = p
            break

    if not gpl_path:
        log_step("STAGE-3: PROBE-MAP", "GPL570.txt not found locally. Fetching via GEOparse...")
        try:
            import GEOparse
            dest = config.paths.dataset_raw
            dest.mkdir(parents=True, exist_ok=True)
            GEOparse.get_GEO("GPL570", destdir=str(dest), silent=True)
            gpl_path = dest / "GPL570.txt"
        except Exception as e:
            raise FileNotFoundError(f"Failed to find or download GPL570 annotation: {e}")

    log_step("STAGE-3: PROBE-MAP", f"Loading GSE15932 probe annotations from {gpl_path.name}...")
    probe_map = {}
    with open(gpl_path, "r", errors="ignore") as f:
        in_table = False
        id_idx, sym_idx = -1, -1
        for line in f:
            if line.startswith("ID\t"):
                in_table = True
                cols = line.strip().split("\t")
                id_idx = cols.index("ID")
                sym_idx = cols.index("Gene Symbol") if "Gene Symbol" in cols else -1
                continue

            if in_table:
                if line.startswith("!"):
                    break
                parts = line.strip().split("\t")
                if len(parts) > max(id_idx, sym_idx) and sym_idx != -1:
                    pid = parts[id_idx].strip()
                    raw_sym = parts[sym_idx].strip()
                    if raw_sym:
                        # Extract first / primary gene symbol
                        primary_sym = raw_sym.split("///")[0].strip()
                        if primary_sym and primary_sym != "---":
                            probe_map[pid] = primary_sym

    log_step("STAGE-3: PROBE-MAP", f"GSE15932: Loaded mapping for {len(probe_map)} probes")
    return probe_map


def map_probes_to_genes(
    df: pd.DataFrame,
    probe_map: Dict[str, str],
    method: str = "mean"
) -> pd.DataFrame:
    """
    Maps probe IDs to HGNC Gene Symbols and aggregates multiple probes per gene.
    Aggregation methods:
    - 'mean': Arithmetic average across probes (standard limma/avereps style)
    - 'median': Median probe expression
    - 'max_mean': Selects the probe with the highest mean expression across samples (WGCNA style)
    """
    mapped_series = df.index.map(probe_map)
    valid_mask = mapped_series.notna() & (mapped_series != "") & (mapped_series != "---")
    
    n_total = len(df)
    n_mapped = valid_mask.sum()
    log_step("STAGE-3: AGG", f"Mapping probes to symbols: {n_mapped}/{n_total} probes mapped ({n_total - n_mapped} unmapped dropped)")

    df_filtered = df[valid_mask].copy()
    symbols = mapped_series[valid_mask].values

    if method == "mean":
        df_filtered["gene_symbol"] = symbols
        aggregated = df_filtered.groupby("gene_symbol").mean()
    elif method == "median":
        df_filtered["gene_symbol"] = symbols
        aggregated = df_filtered.groupby("gene_symbol").median()
    elif method == "max_mean":
        sample_cols = list(df_filtered.columns)
        df_filtered["gene_symbol"] = symbols
        df_filtered["_probe_mean"] = df_filtered[sample_cols].mean(axis=1)
        idx_max = df_filtered.groupby("gene_symbol")["_probe_mean"].idxmax()
        aggregated = df_filtered.loc[idx_max, sample_cols + ["gene_symbol"]].set_index("gene_symbol")
    else:
        raise ValueError(f"Unknown probe aggregation method: '{method}'. Choose 'mean', 'median', or 'max_mean'.")

    aggregated.index.name = "gene_symbol"
    log_step("STAGE-3: AGG", f"Collapsed to {aggregated.shape[0]} unique HGNC gene symbols using '{method}'")
    return aggregated


def preprocess_microarray_cohort(
    cohort_name: str,
    df_raw: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> pd.DataFrame:
    """
    Executes full Stage 3 preprocessing for a microarray cohort:
    1. Log2 transformation (with offset and clipping)
    2. Quantile normalization
    3. Probe mapping and aggregation to HGNC Gene Symbols
    4. Persists processed matrix to disk
    """
    cohort_cfg = config.cohorts[cohort_name]
    norm_cfg = config.normalization

    log_step(f"STAGE-3: {cohort_name}", f"Starting Microarray Preprocessing Pipeline ({df_raw.shape[0]} probes x {df_raw.shape[1]} samples)")

    # 1. Log2 Transformation
    df_transformed = df_raw
    if norm_cfg.microarray_log2_transform:
        df_transformed = log2_transform_microarray(df_raw, offset=norm_cfg.microarray_offset)

    # 2. Quantile Normalization
    df_normalized = df_transformed
    if norm_cfg.microarray_quantile_normalize:
        df_normalized = quantile_normalize(df_transformed)

    # 3. Probe Mapping
    if cohort_name == "GSE26168":
        probe_map = load_gse26168_probe_map(config)
    elif cohort_name == "GSE15932":
        probe_map = load_gse15932_probe_map(config)
    else:
        raise ValueError(f"No probe mapping configuration registered for cohort '{cohort_name}'")

    df_gene = map_probes_to_genes(
        df_normalized,
        probe_map,
        method=norm_cfg.probe_aggregation_method
    )

    # 4. Save to Disk
    dest_path = config.paths.data_processed / cohort_cfg.normalized_output_filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df_gene.to_csv(dest_path, compression="gzip")
    log_step(f"STAGE-3: {cohort_name}", f"Saved normalized gene matrix to {dest_path.name} ({df_gene.shape[0]} genes x {df_gene.shape[1]} samples)")

    return df_gene


# --- RNA-Seq Filtering, DESeq2 Size-Factors & CPM Logic ---

def filter_low_counts(
    df: pd.DataFrame,
    min_total_counts: int = 10,
    min_samples_expressing: int = 5,
    min_count: int = 1
) -> pd.DataFrame:
    """
    Filters out lowly expressed / non-informative genes in RNA-Seq data.
    Retains genes with:
    - Total sum across all samples >= min_total_counts
    - Expression (counts >= min_count) in at least min_samples_expressing samples
    """
    n_before = len(df)
    total_mask = df.sum(axis=1) >= min_total_counts
    sample_mask = (df >= min_count).sum(axis=1) >= min_samples_expressing
    keep_mask = total_mask & sample_mask

    df_filtered = df[keep_mask].copy()
    n_after = len(df_filtered)
    log_step(
        "STAGE-3: RNA-FILTER",
        f"Filtered low counts: {n_after}/{n_before} genes retained ({n_before - n_after} removed, "
        f"criteria: total >= {min_total_counts} and count >= {min_count} in >= {min_samples_expressing} samples)"
    )
    return df_filtered


def calculate_deseq2_size_factors(df: pd.DataFrame) -> pd.Series:
    """
    Calculates DESeq2 median-of-ratios size factors for each sample:
    1. Computes geometric mean of each gene across all samples.
    2. Computes the ratio of each gene's count in a sample to its geometric mean.
    3. Size factor for each sample is the median ratio across all eligible genes.
    Falls back to poscounts (geometric mean over positive counts) if all-positive genes are sparse.
    """
    # Standard DESeq2 considers genes with non-zero counts across all samples
    all_positive = (df > 0).all(axis=1)
    n_all_pos = all_positive.sum()

    if n_all_pos >= 100:
        sub = df[all_positive]
        log_geom_means = np.log(sub).mean(axis=1)
        geom_means = np.exp(log_geom_means)
        ratios = sub.divide(geom_means, axis=0)
        size_factors = ratios.median(axis=0)
        log_step("STAGE-3: DESEQ2-SF", f"Computed standard size factors using {n_all_pos} all-positive genes (range: {size_factors.min():.3f} - {size_factors.max():.3f})")
    else:
        # Poscounts method: compute geometric mean per gene using only positive counts
        log_step("STAGE-3: DESEQ2-SF", "Using poscounts geometric mean estimation for sparse samples...")
        ratios_dict = {}
        for col in df.columns:
            gene_geom = df.apply(lambda row: np.exp(np.mean(np.log(row[row > 0]))) if (row > 0).any() else 0.0, axis=1)
            valid = (df[col] > 0) & (gene_geom > 0)
            sample_ratios = df.loc[valid, col] / gene_geom[valid]
            ratios_dict[col] = sample_ratios.median() if len(sample_ratios) > 0 else 1.0
        size_factors = pd.Series(ratios_dict)

    # Normalize size factors so their geometric mean is 1.0 (standard DESeq2 convention)
    sf_geom = np.exp(np.log(size_factors).mean())
    if sf_geom > 0:
        size_factors = size_factors / sf_geom

    return size_factors


def normalize_deseq2(
    df: pd.DataFrame,
    size_factors: Optional[pd.Series] = None,
    pseudo_count: float = 1.0
) -> pd.DataFrame:
    """
    Normalizes RNA-Seq count matrix by DESeq2 size-factors:
    normalized_counts = raw_counts / size_factors
    Returns log2(normalized_counts + pseudo_count).
    """
    if size_factors is None:
        size_factors = calculate_deseq2_size_factors(df)

    norm_counts = df.divide(size_factors, axis=1)
    norm_log2 = np.log2(norm_counts + pseudo_count)
    log_step("STAGE-3: DESEQ2-NORM", f"Normalized by DESeq2 size-factors + log2 (pseudo_count={pseudo_count})")
    return norm_log2


def calculate_cpm_log2(df: pd.DataFrame, pseudo_count: float = 1.0) -> pd.DataFrame:
    """
    Calculates Counts Per Million (CPM) and applies log2 transformation:
    CPM = (raw_counts / library_size) * 1e6
    Returns log2(CPM + pseudo_count).
    """
    library_sizes = df.sum(axis=0)
    cpm = df.divide(library_sizes, axis=1) * 1e6
    cpm_log2 = np.log2(cpm + pseudo_count)
    log_step("STAGE-3: CPM-LOG2", f"Calculated CPM log2 transformation (pseudo_count={pseudo_count}, mean lib size: {library_sizes.mean():,.0f})")
    return cpm_log2


def preprocess_rnaseq_cohort(
    df_raw: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Dict[str, pd.DataFrame]:
    """
    Executes full Stage 3 preprocessing for RNA-Seq cohort GSE164416:
    1. Low-count filtering based on PreprocessingConfig
    2. DESeq2 size-factor estimation and log2 transformation
    3. CPM scaling and log2 transformation
    4. Persists primary normalized matrix (based on rnaseq_scaling_method) and alternate outputs
    """
    cohort_cfg = config.cohorts["GSE164416"]
    norm_cfg = config.normalization

    log_step("STAGE-3: GSE164416", f"Starting RNA-Seq Preprocessing Pipeline ({df_raw.shape[0]} genes x {df_raw.shape[1]} samples)")

    # 1. Filter low counts
    df_filtered = filter_low_counts(
        df_raw,
        min_total_counts=norm_cfg.rnaseq_min_total_counts,
        min_samples_expressing=norm_cfg.rnaseq_min_samples_expressing,
        min_count=1
    )

    # 2. DESeq2 Size Factor Normalization
    size_factors = calculate_deseq2_size_factors(df_filtered)
    df_deseq2_log2 = normalize_deseq2(df_filtered, size_factors=size_factors, pseudo_count=norm_cfg.rnaseq_pseudo_count)

    # 3. CPM log2 transformation
    df_cpm_log2 = calculate_cpm_log2(df_filtered, pseudo_count=norm_cfg.rnaseq_pseudo_count)

    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    # Save size factors table
    sf_path = dest / "GSE164416_deseq2_size_factors.csv"
    size_factors.to_frame(name="size_factor").to_csv(sf_path)

    # Save both representations for downstream flexibility
    deseq2_path = dest / "GSE164416_deseq2_normalized.csv.gz"
    df_deseq2_log2.to_csv(deseq2_path, compression="gzip")

    cpm_path = dest / "GSE164416_log2_cpm.csv.gz"
    df_cpm_log2.to_csv(cpm_path, compression="gzip")

    # Select primary normalized output according to config
    primary_method = norm_cfg.rnaseq_scaling_method.lower()
    if primary_method in ["deseq2", "deseq_median"]:
        primary_df = df_deseq2_log2
    else:  # Default 'log2_cpm'
        primary_df = df_cpm_log2

    primary_out = dest / cohort_cfg.normalized_output_filename
    primary_df.to_csv(primary_out, compression="gzip")

    log_step("STAGE-3: GSE164416", f"Primary normalized counts saved to {primary_out.name} ({primary_df.shape[0]} genes x {primary_df.shape[1]} samples, scaling={primary_method})")
    log_step("STAGE-3: GSE164416", f"DESeq2 normalized matrix saved to {deseq2_path.name}")
    log_step("STAGE-3: GSE164416", f"Log2-CPM normalized matrix saved to {cpm_path.name}")

    return {
        "primary": primary_df,
        "deseq2": df_deseq2_log2,
        "log2_cpm": df_cpm_log2,
        "size_factors": size_factors.to_frame(name="size_factor")
    }


# --- Stage 3 Orchestrator & Output Validation ---

def run_stage_3_normalization(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    raw_matrices: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Executes Stage 3 across all three cohorts:
    - GSE26168 (Microarray): log2 -> quantile norm -> HGNC mapping
    - GSE15932 (Microarray): log2 -> quantile norm -> HGNC mapping
    - GSE164416 (RNA-Seq): low count filter -> DESeq2 size-factors / log2 CPM
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-3", "Starting Stage 3 Normalization & Preprocessing Logic...")
    normalized_results = {}

    # Load or reuse raw matrices
    def _get_raw(cohort_name: str) -> pd.DataFrame:
        if raw_matrices and cohort_name in raw_matrices:
            return raw_matrices[cohort_name]
        raw_path = dest / config.cohorts[cohort_name].raw_output_filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw expression matrix {raw_path.name} not found. Run Stage 1 first.")
        log_step(f"STAGE-3: LOAD", f"Loading raw matrix for {cohort_name} from {raw_path.name}...")
        return pd.read_csv(raw_path, compression="gzip", index_col=0)

    # 1. GSE26168 (Microarray)
    df_26168_raw = _get_raw("GSE26168")
    df_26168_norm = preprocess_microarray_cohort("GSE26168", df_26168_raw, config)
    normalized_results["GSE26168"] = df_26168_norm

    # 2. GSE15932 (Microarray)
    df_15932_raw = _get_raw("GSE15932")
    df_15932_norm = preprocess_microarray_cohort("GSE15932", df_15932_raw, config)
    normalized_results["GSE15932"] = df_15932_norm

    # 3. GSE164416 (RNA-Seq)
    df_164416_raw = _get_raw("GSE164416")
    rnaseq_results = preprocess_rnaseq_cohort(df_164416_raw, config)
    normalized_results["GSE164416"] = rnaseq_results["primary"]

    log_step("STAGE-3", "Completed Stage 3 Normalization & Preprocessing across all 3 cohorts.")
    return normalized_results


def validate_stage_3_outputs(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> bool:
    """
    Validates Stage 3 outputs:
    - Verifies file existence for all normalized matrices
    - Verifies non-empty shape, zero NaN / Inf values
    - Validates sample column alignment with unified metadata
    """
    dest = config.paths.data_processed
    meta_path = dest / config.metadata.unified_metadata_filename
    meta_df = pd.read_csv(meta_path) if meta_path.exists() else None

    log_step("STAGE-3: VERIFY", "Verifying Stage 3 output matrices integrity...")

    all_passed = True
    for cohort_name, cohort_cfg in config.cohorts.items():
        out_path = dest / cohort_cfg.normalized_output_filename
        if not out_path.exists():
            log_step("STAGE-3: VERIFY", f"[FAIL] Missing {out_path.name}")
            all_passed = False
            continue

        df = pd.read_csv(out_path, compression="gzip", index_col=0)
        n_features, n_samples = df.shape

        # NaN / Inf checks
        n_nans = df.isna().sum().sum()
        n_infs = np.isinf(df.values).sum()
        min_v = df.values.min()
        max_v = df.values.max()

        if n_nans > 0 or n_infs > 0:
            log_step("STAGE-3: VERIFY", f"[FAIL] {cohort_name}: {n_nans} NaNs, {n_infs} Infs found!")
            all_passed = False
            continue

        # Alignment check against clinical metadata
        if meta_df is not None:
            cohort_samples = set(meta_df[meta_df["cohort"] == cohort_name]["sample_id"])
            matrix_samples = set(df.columns)
            if cohort_samples != matrix_samples:
                log_step("STAGE-3: VERIFY", f"[FAIL] {cohort_name}: Sample columns do not match metadata!")
                all_passed = False
                continue

        log_step(
            "STAGE-3: VERIFY",
            f"[PASS] {cohort_name} ({cohort_cfg.modality}): {n_features} features x {n_samples} samples | "
            f"range: [{min_v:.2f}, {max_v:.2f}] | 0 NaNs | 100% sample metadata alignment"
        )

    return all_passed


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG) -> Dict[str, Any]:
    """
    End-to-end Pipeline Orchestrator executing:
    - Stage 1: Cohort expression extraction
    - Stage 2: Clinical metadata harmonization & alignment validation
    - Stage 3: Cohort-specific normalization & preprocessing logic
    - Final output verification
    """
    log_step("PIPELINE", "Initiating Full Preprocessing Pipeline (Stages 1-3) with PreprocessingConfig")
    # Save active config JSON
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

    if alignment_ok and stage_3_ok:
        log_step("PIPELINE", "SUCCESS: Pipeline Stages 1, 2, and 3 completed and verified with 0 errors.")
    else:
        log_step("PIPELINE: WARNING", "Pipeline finished with validation warnings.")

    return {
        "metadata": unified_meta,
        "normalized_matrices": normalized_matrices
    }


if __name__ == "__main__":
    run_pipeline()
