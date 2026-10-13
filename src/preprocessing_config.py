"""
Preprocessing Configuration Module
Centralized settings, cohort definitions, paths, normalization parameters,
and clinical schemas for the data preprocessing pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json

# Project Root Directory
PROJECT_ROOT = Path(r"C:/Bioinformatics Project")


@dataclass
class PathSettings:
    project_root: Path = PROJECT_ROOT
    dataset_raw: Path = PROJECT_ROOT / "dataset" / "raw"
    data_raw_alt: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"


@dataclass
class CohortSettings:
    name: str
    modality: str  # 'Microarray' or 'RNA-Seq'
    platform_id: str
    tissue_source: str
    raw_subfolder: str
    expected_patterns: List[str]
    series_matrix_pattern: Optional[str]
    raw_output_filename: str
    normalized_output_filename: str


@dataclass
class NormalizationSettings:
    # Microarray settings
    microarray_log2_transform: bool = True
    microarray_quantile_normalize: bool = True
    microarray_offset: float = 1.0  # Added before log2 if values are raw intensity

    # RNA-Seq settings
    rnaseq_min_total_counts: int = 10
    rnaseq_min_samples_expressing: int = 5
    rnaseq_scaling_method: str = "log2_cpm"  # Options: 'log2_cpm', 'tpm', 'deseq_median'
    rnaseq_pseudo_count: float = 1.0


@dataclass
class MetadataSettings:
    unified_metadata_filename: str = "unified_clinical_metadata.csv"
    standard_columns: List[str] = field(default_factory=lambda: [
        "sample_id",
        "cohort",
        "platform_type",
        "platform_id",
        "tissue_source",
        "donor_identifier",
        "clinical_group",
        "diabetes_status",
        "cancer_status",
        "sex",
        "age_years",
        "fbs_mg_dl",
        "raw_title",
        "raw_characteristics"
    ])


@dataclass
class PreprocessingConfig:
    paths: PathSettings = field(default_factory=PathSettings)
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    cohorts: Dict[str, CohortSettings] = field(default_factory=lambda: {
        "GSE26168": CohortSettings(
            name="GSE26168",
            modality="Microarray",
            platform_id="GPL6883",
            tissue_source="Whole Blood",
            raw_subfolder="GSE26168_RAW",
            expected_patterns=["*.bgx", "*.bgx.gz", "*series_matrix*.txt.gz"],
            series_matrix_pattern="*GPL6883*series_matrix*.txt.gz",
            raw_output_filename="GSE26168_raw_expression.csv.gz",
            normalized_output_filename="GSE26168_normalized_expression.csv.gz"
        ),
        "GSE15932": CohortSettings(
            name="GSE15932",
            modality="Microarray",
            platform_id="GPL570",
            tissue_source="PBMC",
            raw_subfolder="GSE15932_RAW",
            expected_patterns=["*.CEL", "*.CEL.gz", "*.CHP", "*.CHP.gz", "*series_matrix*.txt.gz"],
            series_matrix_pattern="*series_matrix*.txt.gz",
            raw_output_filename="GSE15932_raw_expression.csv.gz",
            normalized_output_filename="GSE15932_normalized_expression.csv.gz"
        ),
        "GSE164416": CohortSettings(
            name="GSE164416",
            modality="RNA-Seq",
            platform_id="Illumina HiSeq (HTSeq)",
            tissue_source="Pancreatic Islets",
            raw_subfolder="GSE164416_RAW",
            expected_patterns=["*.txt.gz", "*_htseq_counts.txt.gz"],
            series_matrix_pattern=None,
            raw_output_filename="GSE164416_raw_counts.csv.gz",
            normalized_output_filename="GSE164416_normalized_counts.csv.gz"
        )
    })

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to JSON-serializable dictionary."""
        d = asdict(self)
        # Convert Paths to strings
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d

    def save_json(self, output_path: Optional[Path] = None):
        """Persists current configuration to JSON file."""
        target = output_path or (self.paths.data_processed / "preprocessing_config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def resolve_raw_cohort_folder(self, cohort_name: str) -> Path:
        """Resolves raw directory with fallback between dataset/raw and data/raw."""
        cohort = self.cohorts[cohort_name]
        primary = self.paths.dataset_raw / cohort.raw_subfolder
        fallback = self.paths.data_raw_alt / cohort.raw_subfolder
        if not primary.exists() and fallback.exists():
            return fallback
        return primary


# Global default configuration instance
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()


if __name__ == "__main__":
    cfg = DEFAULT_PREPROCESSING_CONFIG
    cfg.save_json()
    print(f"[CONFIG] Preprocessing configuration validated and saved to {cfg.paths.data_processed / 'preprocessing_config.json'}")
