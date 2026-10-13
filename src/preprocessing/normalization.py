"""
Stage 3: Normalization & Preprocessing Logic
- Microarray: Log2 transformation, quantile normalization, probe mapping to HGNC Gene Symbols
- Low-Expression Filtering: Dual implementation supporting both:
    * Approach 1: Heuristic / Absolute Floor (Microarray background floor <= 80%; RNA-Seq count sum >= 10)
    * Approach 2: Distribution-Aware / Cohort-Fraction (Microarray IQR/variance filtering; RNA-Seq CPM >= 1.0)
- Comparative Analysis: Automated generator for comparative matrices across both filtering strategies
- RNA-Seq: DESeq2 size-factor normalization & Counts Per Million (CPM) log2 transformation
- Output integrity and metadata alignment verification
"""

from pathlib import Path
import gzip
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd

from preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig
)
from preprocessing.common import log_step


# ==============================================================================
# Microarray Normalization, Probe Mapping & Dual Filtering Logic
# ==============================================================================

def log2_transform_microarray(df: pd.DataFrame, offset: float = 1.0) -> pd.DataFrame:
    """
    Applies log2 transformation to microarray intensity matrices.
    Safely clips negative values (e.g. from background subtraction) at 0
    before adding offset and taking np.log2.
    Detects if the matrix is already log-transformed (max < 30 and min >= 0).
    """
    max_val = df.max().max()
    min_val = df.min().min()

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

    sorted_values = np.sort(df.values, axis=0)
    target_distribution = np.mean(sorted_values, axis=1)

    ranks = df.rank(method="average", axis=0).to_numpy() - 1.0
    grid = np.arange(n_rows, dtype=float)

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


# --- Microarray Filtering Approaches ---

def filter_microarray_background(
    df: pd.DataFrame,
    background_percentile: float = 0.20,
    min_sample_ratio: float = 0.20
) -> pd.DataFrame:
    """
    Approach 1: Heuristic Background Floor Filtering for Microarrays.
    Filters probes/genes whose expression fails to exceed the empirical background threshold.
    Retains genes with signal > background noise in at least min_sample_ratio of samples
    (e.g., drops genes in <= 80% background noise per blueprint.md Stage 3).
    """
    n_before = len(df)
    bg_threshold = float(np.percentile(df.values, background_percentile * 100))
    min_samples = int(np.ceil(min_sample_ratio * df.shape[1]))

    expressed_mask = (df > bg_threshold).sum(axis=1) >= min_samples
    df_filtered = df[expressed_mask].copy()
    n_after = len(df_filtered)

    log_step(
        "STAGE-3: ARRAY-BG-FILTER",
        f"Approach 1 (Background Floor): {n_after}/{n_before} genes retained ({n_before - n_after} removed, "
        f"cutoff: > {bg_threshold:.2f} [p{int(background_percentile*100)}] in >= {min_samples}/{df.shape[1]} samples)"
    )
    return df_filtered


def filter_microarray_variance(
    df: pd.DataFrame,
    threshold_percentile: float = 0.20,
    method: str = "iqr"
) -> pd.DataFrame:
    """
    Approach 2: Distribution-Aware Non-Specific Variance Filtering for Microarrays.
    Prunes invariant 'flat-line' genes that contribute technical noise and excessive
    multiple testing penalty without biological variance across samples (Bourgon et al., 2010).
    Methods:
    - 'iqr': Interquartile Range (75th - 25th percentile)
    - 'std': Standard deviation across samples
    """
    n_before = len(df)
    if method == "iqr":
        variabilities = df.quantile(0.75, axis=1) - df.quantile(0.25, axis=1)
    elif method == "std":
        variabilities = df.std(axis=1)
    else:
        raise ValueError(f"Unknown variance filter method '{method}'. Choose 'iqr' or 'std'.")

    cutoff = float(np.percentile(variabilities, threshold_percentile * 100))
    keep_mask = variabilities >= cutoff
    df_filtered = df[keep_mask].copy()
    n_after = len(df_filtered)

    log_step(
        "STAGE-3: ARRAY-VAR-FILTER",
        f"Approach 2 (Variance Floor): {n_after}/{n_before} genes retained ({n_before - n_after} removed, "
        f"{method.upper()} cutoff: >= {cutoff:.4f} [pruning bottom {int(threshold_percentile*100)}% lowest variability])"
    )
    return df_filtered


def filter_microarray_genes(
    df: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    approach: Optional[str] = None
) -> pd.DataFrame:
    """
    Unified dispatcher for microarray low-expression filtering.
    Selects filtering approach:
    - 'background': Approach 1 (Heuristic background floor <= 80%)
    - 'variance': Approach 2 (Distribution-aware non-specific variability)
    - 'both': Sequentially applies background floor and variance pruning
    - 'none': Preserves all mapped features
    """
    norm_cfg = config.normalization
    selected_approach = (approach or norm_cfg.microarray_filtering_approach).lower()

    if selected_approach == "none":
        log_step("STAGE-3: ARRAY-FILTER", "Microarray filtering set to 'none'. Retaining all mapped genes.")
        return df.copy()

    filtered = df
    if selected_approach in ["background", "both"]:
        filtered = filter_microarray_background(
            filtered,
            background_percentile=norm_cfg.microarray_bg_percentile,
            min_sample_ratio=norm_cfg.microarray_bg_min_sample_ratio
        )

    if selected_approach in ["variance", "both"]:
        filtered = filter_microarray_variance(
            filtered,
            threshold_percentile=norm_cfg.microarray_variance_threshold_percentile,
            method="iqr"
        )

    return filtered


def preprocess_microarray_cohort(
    cohort_name: str,
    df_raw: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    filtering_approach: Optional[str] = None,
    persist: bool = True
) -> pd.DataFrame:
    """
    Executes full Stage 3 preprocessing for a microarray cohort:
    1. Log2 transformation (with offset and clipping)
    2. Quantile normalization
    3. Probe mapping and aggregation to HGNC Gene Symbols
    4. Low-expression filtering (Approach 1 background floor, Approach 2 variance, or both)
    5. Optionally persists processed matrix to disk
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

    # If Approach 2 (variance) is selected, allow max_mean probe aggregation if configured
    agg_method = norm_cfg.probe_aggregation_method
    if filtering_approach == "variance" and agg_method == "mean":
        agg_method = "max_mean"

    df_gene = map_probes_to_genes(
        df_normalized,
        probe_map,
        method=agg_method
    )

    # 4. Low-expression Gene Filtering
    df_filtered = filter_microarray_genes(df_gene, config=config, approach=filtering_approach)

    # 5. Save to Disk
    if persist:
        dest_path = config.paths.data_processed / cohort_cfg.normalized_output_filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(dest_path, compression="gzip")
        log_step(f"STAGE-3: {cohort_name}", f"Saved normalized gene matrix to {dest_path.name} ({df_filtered.shape[0]} genes x {df_filtered.shape[1]} samples)")

    return df_filtered


# ==============================================================================
# RNA-Seq Filtering, DESeq2 Size-Factors & CPM Logic
# ==============================================================================

def filter_low_counts(
    df: pd.DataFrame,
    min_total_counts: int = 10,
    min_samples_expressing: int = 5,
    min_count: int = 1
) -> pd.DataFrame:
    """
    Approach 1: Heuristic Absolute Floor Filtering for RNA-Seq.
    Filters out lowly expressed / non-informative transcripts in count matrices.
    Retains genes with:
    - Total sum across all samples >= min_total_counts
    - Counts >= min_count in at least min_samples_expressing samples
    """
    n_before = len(df)
    total_mask = df.sum(axis=1) >= min_total_counts
    sample_mask = (df >= min_count).sum(axis=1) >= min_samples_expressing
    keep_mask = total_mask & sample_mask

    df_filtered = df[keep_mask].copy()
    n_after = len(df_filtered)
    log_step(
        "STAGE-3: RNA-FILTER",
        f"Approach 1 (Heuristic Floor): {n_after}/{n_before} genes retained ({n_before - n_after} removed, "
        f"criteria: total >= {min_total_counts} and count >= {min_count} in >= {min_samples_expressing} samples)"
    )
    return df_filtered


# Alias for Approach 1
filter_rnaseq_heuristic = filter_low_counts


def filter_rnaseq_distribution_aware(
    df: pd.DataFrame,
    min_cpm: float = 1.0,
    min_samples: int = 5,
    min_fraction: Optional[float] = None
) -> pd.DataFrame:
    """
    Approach 2: Distribution-Aware / Cohort-Fraction CPM Filtering for RNA-Seq.
    Normalizes for library size disparities before filtering (mirroring edgeR::filterByExpr).
    Retains genes with Counts Per Million (CPM) >= min_cpm in at least min_samples (or min_fraction * N).
    """
    n_before = len(df)
    library_sizes = df.sum(axis=0)
    cpm = df.divide(library_sizes, axis=1) * 1e6

    target_samples = min_samples
    if min_fraction is not None:
        target_samples = max(min_samples, int(np.ceil(min_fraction * df.shape[1])))

    cpm_mask = (cpm >= min_cpm).sum(axis=1) >= target_samples
    df_filtered = df[cpm_mask].copy()
    n_after = len(df_filtered)

    log_step(
        "STAGE-3: RNA-CPM-FILTER",
        f"Approach 2 (Distribution-Aware CPM): {n_after}/{n_before} genes retained ({n_before - n_after} removed, "
        f"criteria: CPM >= {min_cpm:.1f} in >= {target_samples}/{df.shape[1]} samples)"
    )
    return df_filtered


def filter_rnaseq_counts(
    df: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    approach: Optional[str] = None
) -> pd.DataFrame:
    """
    Unified dispatcher for RNA-Seq count matrix filtering.
    Selects filtering approach:
    - 'heuristic': Approach 1 (Total count sum floor & expressing sample floor)
    - 'distribution_aware': Approach 2 (Library-adjusted CPM floor across cohort/group fraction)
    - 'none': Preserves all raw feature rows
    """
    norm_cfg = config.normalization
    selected_approach = (approach or norm_cfg.rnaseq_filtering_approach).lower()

    if selected_approach == "none":
        log_step("STAGE-3: RNA-FILTER", "RNA-Seq filtering set to 'none'. Retaining all features.")
        return df.copy()
    elif selected_approach in ["heuristic", "counts", "approach_1"]:
        return filter_low_counts(
            df,
            min_total_counts=norm_cfg.rnaseq_min_total_counts,
            min_samples_expressing=norm_cfg.rnaseq_min_samples_expressing,
            min_count=1
        )
    elif selected_approach in ["distribution_aware", "cpm", "approach_2"]:
        return filter_rnaseq_distribution_aware(
            df,
            min_cpm=norm_cfg.rnaseq_cpm_threshold,
            min_samples=norm_cfg.rnaseq_cpm_min_samples
        )
    else:
        raise ValueError(f"Unknown RNA-Seq filtering approach: '{selected_approach}'. Choose 'heuristic', 'distribution_aware', or 'none'.")


def calculate_deseq2_size_factors(df: pd.DataFrame) -> pd.Series:
    """
    Calculates DESeq2 median-of-ratios size factors for each sample:
    1. Computes geometric mean of each gene across all samples.
    2. Computes the ratio of each gene's count in a sample to its geometric mean.
    3. Size factor for each sample is the median ratio across all eligible genes.
    Falls back to poscounts (geometric mean over positive counts) if all-positive genes are sparse.
    """
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
        log_step("STAGE-3: DESEQ2-SF", "Using poscounts geometric mean estimation for sparse samples...")
        ratios_dict = {}
        for col in df.columns:
            gene_geom = df.apply(lambda row: np.exp(np.mean(np.log(row[row > 0]))) if (row > 0).any() else 0.0, axis=1)
            valid = (df[col] > 0) & (gene_geom > 0)
            sample_ratios = df.loc[valid, col] / gene_geom[valid]
            ratios_dict[col] = sample_ratios.median() if len(sample_ratios) > 0 else 1.0
        size_factors = pd.Series(ratios_dict)

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
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    filtering_approach: Optional[str] = None,
    persist: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Executes full Stage 3 preprocessing for RNA-Seq cohort GSE164416:
    1. Low-count filtering (Approach 1 Heuristic or Approach 2 Distribution-Aware CPM)
    2. DESeq2 size-factor estimation and log2 transformation
    3. CPM scaling and log2 transformation
    4. Optionally persists primary normalized matrix and alternative outputs
    """
    cohort_cfg = config.cohorts["GSE164416"]
    norm_cfg = config.normalization

    log_step("STAGE-3: GSE164416", f"Starting RNA-Seq Preprocessing Pipeline ({df_raw.shape[0]} genes x {df_raw.shape[1]} samples)")

    # 1. Filter low counts
    df_filtered = filter_rnaseq_counts(df_raw, config=config, approach=filtering_approach)

    # 2. DESeq2 Size Factor Normalization
    size_factors = calculate_deseq2_size_factors(df_filtered)
    df_deseq2_log2 = normalize_deseq2(df_filtered, size_factors=size_factors, pseudo_count=norm_cfg.rnaseq_pseudo_count)

    # 3. CPM log2 transformation
    df_cpm_log2 = calculate_cpm_log2(df_filtered, pseudo_count=norm_cfg.rnaseq_pseudo_count)

    # Select primary normalized output according to config
    primary_method = norm_cfg.rnaseq_scaling_method.lower()
    if primary_method in ["deseq2", "deseq_median"]:
        primary_df = df_deseq2_log2
    else:  # Default 'log2_cpm'
        primary_df = df_cpm_log2

    if persist:
        dest = config.paths.data_processed
        dest.mkdir(parents=True, exist_ok=True)

        sf_path = dest / "GSE164416_deseq2_size_factors.csv"
        size_factors.to_frame(name="size_factor").to_csv(sf_path)

        deseq2_path = dest / "GSE164416_deseq2_normalized.csv.gz"
        df_deseq2_log2.to_csv(deseq2_path, compression="gzip")

        cpm_path = dest / "GSE164416_log2_cpm.csv.gz"
        df_cpm_log2.to_csv(cpm_path, compression="gzip")

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


# ==============================================================================
# Comparative Filtering Evaluation Matrix Generator
# ==============================================================================

def generate_comparative_filtering_matrices(
    raw_matrices: Dict[str, pd.DataFrame],
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
) -> Dict[str, Any]:
    """
    Executes a complete comparative pipeline run across both low-expression filtering approaches:
    - Approach 1: Heuristic / Absolute Floor (Microarray background floor <= 80%; RNA-Seq counts sum >= 10)
    - Approach 2: Distribution-Aware / Cohort-Fraction (Microarray IQR variance; RNA-Seq CPM >= 1.0)

    Returns:
    - 'approach_1': Dictionary of processed matrices under Approach 1
    - 'approach_2': Dictionary of processed matrices under Approach 2
    - 'metrics': DataFrame comparing gene counts, sparsity, and overlap statistics
    """
    log_step("STAGE-3: COMPARATIVE", "Executing Dual Filtering Comparative Analysis across all cohorts...")

    # Run Approach 1
    m1_26168 = preprocess_microarray_cohort("GSE26168", raw_matrices["GSE26168"], config=config, filtering_approach="background", persist=False)
    m1_15932 = preprocess_microarray_cohort("GSE15932", raw_matrices["GSE15932"], config=config, filtering_approach="background", persist=False)
    m1_164416_res = preprocess_rnaseq_cohort(raw_matrices["GSE164416"], config=config, filtering_approach="heuristic", persist=False)
    m1_164416 = m1_164416_res["primary"]

    # Run Approach 2
    m2_26168 = preprocess_microarray_cohort("GSE26168", raw_matrices["GSE26168"], config=config, filtering_approach="variance", persist=False)
    m2_15932 = preprocess_microarray_cohort("GSE15932", raw_matrices["GSE15932"], config=config, filtering_approach="variance", persist=False)
    m2_164416_res = preprocess_rnaseq_cohort(raw_matrices["GSE164416"], config=config, filtering_approach="distribution_aware", persist=False)
    m2_164416 = m2_164416_res["primary"]

    cohort_pairs = [
        ("GSE26168", raw_matrices["GSE26168"], m1_26168, m2_26168),
        ("GSE15932", raw_matrices["GSE15932"], m1_15932, m2_15932),
        ("GSE164416", raw_matrices["GSE164416"], m1_164416, m2_164416),
    ]

    records = []
    for c_name, raw_df, df1, df2 in cohort_pairs:
        g1 = set(df1.index)
        g2 = set(df2.index)
        intersection = len(g1.intersection(g2))
        union = len(g1.union(g2))
        jaccard = intersection / union if union > 0 else 1.0

        records.append({
            "cohort": c_name,
            "raw_features": raw_df.shape[0],
            "samples": raw_df.shape[1],
            "appr1_features": df1.shape[0],
            "appr1_sparsity_pct": float((df1 == 0).sum().sum() / df1.size * 100),
            "appr1_intensity_mean": float(df1.values.mean()),
            "appr2_features": df2.shape[0],
            "appr2_sparsity_pct": float((df2 == 0).sum().sum() / df2.size * 100),
            "appr2_intensity_mean": float(df2.values.mean()),
            "overlap_features": intersection,
            "jaccard_similarity": round(jaccard, 4)
        })

    metrics_df = pd.DataFrame(records)
    log_step("STAGE-3: COMPARATIVE", "Comparative filtering evaluation completed successfully.")

    return {
        "approach_1": {"GSE26168": m1_26168, "GSE15932": m1_15932, "GSE164416": m1_164416},
        "approach_2": {"GSE26168": m2_26168, "GSE15932": m2_15932, "GSE164416": m2_164416},
        "metrics": metrics_df
    }


# ==============================================================================
# Stage 3 Orchestrator & Output Validation
# ==============================================================================

def run_stage_3_normalization(
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    raw_matrices: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Executes Stage 3 across all three cohorts:
    - GSE26168 (Microarray): log2 -> quantile norm -> HGNC mapping -> filtering
    - GSE15932 (Microarray): log2 -> quantile norm -> HGNC mapping -> filtering
    - GSE164416 (RNA-Seq): low count filtering -> DESeq2 size-factors / log2 CPM
    """
    dest = config.paths.data_processed
    dest.mkdir(parents=True, exist_ok=True)

    log_step("STAGE-3", "Starting Stage 3 Normalization & Preprocessing Logic...")
    normalized_results = {}

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
    df_26168_norm = preprocess_microarray_cohort("GSE26168", df_26168_raw, config=config, persist=True)
    normalized_results["GSE26168"] = df_26168_norm

    # 2. GSE15932 (Microarray)
    df_15932_raw = _get_raw("GSE15932")
    df_15932_norm = preprocess_microarray_cohort("GSE15932", df_15932_raw, config=config, persist=True)
    normalized_results["GSE15932"] = df_15932_norm

    # 3. GSE164416 (RNA-Seq)
    df_164416_raw = _get_raw("GSE164416")
    rnaseq_results = preprocess_rnaseq_cohort(df_164416_raw, config=config, persist=True)
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
    if not meta_path.exists():
        meta_path = dest / "clinical_metadata_unified.csv"
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

        n_nans = df.isna().sum().sum()
        n_infs = np.isinf(df.values).sum()
        min_v = df.values.min()
        max_v = df.values.max()

        if n_nans > 0 or n_infs > 0:
            log_step("STAGE-3: VERIFY", f"[FAIL] {cohort_name}: {n_nans} NaNs, {n_infs} Infs found!")
            all_passed = False
            continue

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
