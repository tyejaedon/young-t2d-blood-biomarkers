"""
Bioinformatics Data Cleansing & Transcriptomic Metadata Extraction
Enhanced with TokenGuard Clean Logging and Defensive Metadata Validation
"""

from pathlib import Path
import random
import gzip
import re
from typing import Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Seed chosen for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(r'C:/Bioinformatics Project')
DATASET_RAW = PROJECT_ROOT / 'dataset' / 'raw'
ALT_DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

DATASETS = {
    'GSE26168': {
        'folder': DATASET_RAW / 'GSE26168_RAW',
        'profile': 'Human whole-blood microarray transcriptomic profile (Affymetrix/Illumina probe-level data)',
        'expected_patterns': ['*.bgx', '*.CEL', '*.CEL.gz', '*.bgx.gz'],
    },
    'GSE15932': {
        'folder': DATASET_RAW / 'GSE15932_RAW',
        'profile': 'Human peripheral blood mononuclear cell (PBMC) microarray transcriptomic profile',
        'expected_patterns': ['*.CEL', '*.CEL.gz', '*.CHP', '*.CHP.gz'],
    },
    'GSE164416': {
        'folder': DATASET_RAW / 'GSE164416_RAW',
        'profile': 'Human pancreatic islet RNA-seq transcriptomic profile with HTSeq count files',
        'expected_patterns': ['*.txt.gz', '*.txt'],
    },
}


def log_summary(title: str, message: str):
    """Token-efficient logger replacing decorative ASCII headers."""
    print(f"[DATA-CLEANSE] {title}: {message}")


def check_dataset_folder(dataset_name: str, dataset_info: dict) -> dict:
    folder = dataset_info['folder']
    if not folder.exists() and (ALT_DATA_RAW / folder.name).exists():
        folder = ALT_DATA_RAW / folder.name

    dataset_info['resolved_folder'] = folder
    exists = folder.exists()
    file_count = sum(1 for item in folder.rglob('*') if item.is_file()) if exists else 0

    matches = []
    if exists:
        for pattern in dataset_info['expected_patterns']:
            matches.extend(folder.rglob(pattern))

    evidence_files = ', '.join(sorted({p.name for p in matches})[:3]) if matches else 'none found'
    status = 'READY' if exists and file_count > 0 else 'MISSING / EMPTY'

    return {
        'Dataset': dataset_name,
        'Folder': str(folder.relative_to(PROJECT_ROOT)) if folder.is_relative_to(PROJECT_ROOT) else str(folder),
        'Exists': exists,
        'Files': file_count,
        'Transcriptomic profile': dataset_info['profile'],
        'Evidence files': evidence_files,
        'Status': status,
    }


def validate_dataset_folders():
    summary_rows = [check_dataset_folder(name, info) for name, info in DATASETS.items()]
    summary_df = pd.DataFrame(summary_rows)[['Dataset', 'Status', 'Files', 'Evidence files']]
    all_ready = all(row['Status'] == 'READY' for row in summary_rows)
    
    log_summary("Folder Validation", "All ready" if all_ready else "One or more datasets missing/empty")
    print(tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))


def inspect_dataset_files():
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        if not folder.exists():
            continue
        inspect_file_count = sum(1 for item in folder.rglob('*') if item.is_file())
        log_summary(f"Files: {dataset_name}", f"Total files: {inspect_file_count}")


def extract_geo_metadata(matrix_file: Path) -> Dict[str, List[str]]:
    """Extracts GEO metadata keys and values defensively from gzip files."""
    metadata = {}
    if not matrix_file.exists():
        return metadata

    with gzip.open(matrix_file, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('!'):
                parts = line.strip().split('\t')
                key = parts[0]
                if key not in metadata:
                    metadata[key] = []
                metadata[key].extend(parts[1:])
    return metadata


def inspect_metadata_files():
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        if not folder.exists():
            continue
        matrix_files = list(folder.glob('*series_matrix*.txt.gz'))
        if not matrix_files:
            log_summary(f"Metadata: {dataset_name}", "No series matrix files found.")
            continue
        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)
            log_summary(f"Metadata: {dataset_name}", f"{matrix_file.name} - {len(metadata)} total keys")


def discover_biological_samples() -> pd.DataFrame:
    sample_summary = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        gsm_ids = set()
        if folder.exists():
            for file in folder.rglob('*'):
                match = re.search(r'(GSM\d+)', file.name)
                if match:
                    gsm_ids.add(match.group(1))

        sample_summary.append({
            'Dataset': dataset_name,
            'Biological Samples': len(gsm_ids),
            'Sample GSMs': ', '.join(sorted(gsm_ids)[:3]) if gsm_ids else 'None'
        })
    df = pd.DataFrame(sample_summary)
    log_summary("Biological Sample Discovery", f"Discovered {df['Biological Samples'].sum()} total GSM sample references")
    return df


def discover_disease_groups() -> pd.DataFrame:
    disease_summary = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        if not folder.exists():
            continue
        matrix_files = list(folder.glob('*series_matrix*.txt.gz'))
        if not matrix_files:
            disease_summary.append({'Dataset': dataset_name, 'Disease Groups': 'None', 'Group Count': 0})
            continue

        discovered_groups = set()
        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)
            for field, values in metadata.items():
                if 'characteristics' not in field.lower():
                    continue
                for value in values:
                    val_str = str(value).strip()
                    if ':' in val_str:
                        k, phenotype = val_str.split(':', 1)
                        k = k.strip().lower()
                        phenotype = phenotype.strip()
                        if any(term in k for term in ['disease', 'group', 'phenotype', 'patient type', 'condition', 'status']):
                            discovered_groups.add(phenotype)

        disease_summary.append({
            'Dataset': dataset_name,
            'Disease Groups': '; '.join(sorted(discovered_groups)[:5]) if discovered_groups else 'None',
            'Group Count': len(discovered_groups)
        })
    return pd.DataFrame(disease_summary)


def extract_sample_annotations() -> pd.DataFrame:
    annotations = []
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        if not folder.exists():
            continue
        matrix_files = list(folder.glob('*series_matrix*.txt.gz'))
        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)
            accessions = metadata.get('!Sample_geo_accession', [])
            titles = metadata.get('!Sample_title', [])
            characteristics = metadata.get('!Sample_characteristics_ch1', [])
            sample_count = min(len(accessions), len(titles))

            for i in range(sample_count):
                annotations.append({
                    'Dataset': dataset_name,
                    'GSM': accessions[i],
                    'Title': titles[i],
                    'Characteristics': characteristics[i] if i < len(characteristics) else 'N/A'
                })
    return pd.DataFrame(annotations)


def assess_metadata_completeness() -> pd.DataFrame:
    assessment = []
    required_fields = [
        '!Sample_title',
        '!Sample_geo_accession',
        '!Sample_source_name_ch1',
        '!Sample_characteristics_ch1'
    ]
    for dataset_name, dataset_info in DATASETS.items():
        folder = dataset_info.get('resolved_folder', dataset_info['folder'])
        if not folder.exists():
            continue
        matrix_files = list(folder.glob('*series_matrix*.txt.gz'))
        if not matrix_files:
            assessment.append({'Dataset': dataset_name, 'Missing Fields': len(required_fields), 'Status': 'NO_MATRIX'})
            continue
        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)
            missing = [field for field in required_fields if field not in metadata]
            assessment.append({
                'Dataset': dataset_name,
                'Missing Fields': len(missing),
                'Status': 'COMPLETE' if not missing else 'INCOMPLETE'
            })
    return pd.DataFrame(assessment)

def export_dataframes_to_csv():
    """Exports key dataframes to CSV files for downstream analysis."""
    samples_df = discover_biological_samples()
    groups_df = discover_disease_groups()
    completeness_df = assess_metadata_completeness()
    annotations_df = extract_sample_annotations()

    export_folder = PROJECT_ROOT / 'data' / 'processed'
    export_folder.mkdir(parents=True, exist_ok=True)

    samples_df.to_csv(export_folder / 'biological_samples_summary.csv', index=False)
    groups_df.to_csv(export_folder / 'disease_groups_summary.csv', index=False)
    completeness_df.to_csv(export_folder / 'metadata_completeness_summary.csv', index=False)
    annotations_df.to_csv(export_folder / 'sample_annotations.csv', index=False)

    log_summary("Data Export", f"Exported summaries to {export_folder}")

def main():
    log_summary("Pipeline Start", "Initiating optimized transcriptomic data cleansing checks")
    validate_dataset_folders()
    inspect_dataset_files()
    inspect_metadata_files()
    
    samples_df = discover_biological_samples()
    print(tabulate(samples_df, headers='keys', tablefmt='psql', showindex=False))
    
    groups_df = discover_disease_groups()
    print(tabulate(groups_df, headers='keys', tablefmt='psql', showindex=False))
    
    completeness_df = assess_metadata_completeness()
    print(tabulate(completeness_df, headers='keys', tablefmt='psql', showindex=False))
    
    annotations_df = extract_sample_annotations()
    log_summary("Sample Integrity", f"Total annotations compiled: {len(annotations_df)}")
    log_summary("Pipeline End", "Execution successfully completed")


if __name__ == '__main__':
    main()
