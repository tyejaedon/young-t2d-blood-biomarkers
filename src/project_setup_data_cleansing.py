## This file is used to set up the data cleansing project by creating necessary directories and files.

from pathlib import Path
import random
import gzip
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tabulate import tabulate

## Seed 42 chosen for reproducibility
## project variables
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

## variable holding path to dataset
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


def validate_dataset_folders():
    summary_rows = [
        check_dataset_folder(name, info)
        for name, info in DATASETS.items()
    ]

    summary_df = pd.DataFrame(summary_rows)[
        [
            'Dataset',
            'Status',
            'Files',
            'Transcriptomic profile',
            'Evidence files',
            'Folder'
        ]
    ]

    all_ready = all(row['Status'] == 'READY' for row in summary_rows)

    print('Checking dataset folders and files...')
    display(summary_df.style.hide(axis='index'))

    print(
        'All dataset folders are ready.'
        if all_ready
        else 'One or more dataset folders are missing or empty.'
    )


def inspect_dataset_files():
    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['resolved_folder']

        print(f'\nInspecting files in {dataset_name} ({folder})...')

        inspect_file_count = sum(1 for item in folder.rglob('*') if item.is_file())

        print(f'Total files: {inspect_file_count}')

        for pattern in dataset_info['expected_patterns']:

            matches = list(folder.rglob(pattern))

            match_count = len(matches)

            print(f'Pattern "{pattern}": {match_count} matches')

            if match_count > 0:
                sample_files = sorted({path.name for path in matches})[:5]
                print('Sample files:', ', '.join(sample_files))
            else:
                print('No files found for this pattern.')


## function that performs validation check of folder existenceq
def check_dataset_folder(dataset_name: str, dataset_info: dict) -> dict:
    folder = dataset_info['folder']

    if not folder.exists() and (ALT_DATA_RAW / folder.name).exists():
        folder = ALT_DATA_RAW / folder.name

    dataset_info['resolved_folder'] = folder

    exists = folder.exists()

    file_count = (
        sum(1 for item in folder.rglob('*') if item.is_file())
        if exists else 0
    )

    matches = []

    if exists:
        for pattern in dataset_info['expected_patterns']:
            matches.extend(folder.rglob(pattern))

    if matches:
        sample_files = sorted({path.name for path in matches})[:5]
        evidence_files = ', '.join(sample_files)
    else:
        evidence_files = 'none found for expected file patterns'

    status = 'READY' if exists and file_count > 0 else 'MISSING / EMPTY'

    return {
        'Dataset': dataset_name,
        'Folder': str(folder.relative_to(PROJECT_ROOT))
        if folder.is_relative_to(PROJECT_ROOT)
        else str(folder),
        'Exists': exists,
        'Files': file_count,
        'Transcriptomic profile': dataset_info['profile'],
        'Evidence files': evidence_files,
        'Status': status,
    }


def inspect_metadata_files():
    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        print(f'\nInspecting metadata files in {dataset_name} ({folder})...')

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        if not matrix_files:
            print('No series matrix files found.')
            continue

        for matrix_file in matrix_files:

            print(f'\nReading metadata from {matrix_file.name}...')

            metadata = extract_geo_metadata(matrix_file)
            tabulate(metadata, headers='keys', tablefmt='grid')
            print("\n")

            print(f'Total metadata fields: {len(metadata)}')

            sample_fields = sorted(metadata.keys())[:10]
            tabulate(sample_fields, headers='keys', tablefmt='grid')


## extracts GEO metadata from series matrix files
def extract_geo_metadata(matrix_file):
    metadata = {}

    with gzip.open(matrix_file, 'rt', errors='ignore') as f:

        for line in f:

            if line.startswith('!'):

                parts = line.strip().split('\t')

                key = parts[0]

                if key not in metadata:
                    metadata[key] = []

                metadata[key].extend(parts[1:])
    tabulate(metadata, headers='keys', tablefmt='grid')
    tabulate(metadata.items(), headers=['Field', 'Values'], tablefmt='grid')
    return metadata


## extracts biological sample information from GEO datasets
def discover_biological_samples():
    import re

    sample_summary = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        print(f"\n{'=' * 80}")
        print(f'Biological Sample Discovery : {dataset_name}')
        print(f"{'=' * 80}")

        gsm_ids = set()

        for file in folder.rglob('*'):

            match = re.search(r'(GSM\d+)', file.name)

            if match:
                gsm_ids.add(match.group(1))

        sample_count = len(gsm_ids)

        print(f'Unique biological samples: {sample_count}')

        if gsm_ids:

            print('\nExample GSM identifiers:')

            for gsm in sorted(gsm_ids)[:10]:
                print(gsm)


        sample_summary.append({
            'Dataset': dataset_name,
            'Biological Samples': sample_count
        })
        tabulate(sample_summary, headers='keys', tablefmt='grid')



    return pd.DataFrame(sample_summary)

## extracts disease and phenotype groups from GEO metadata
def discover_disease_groups():

    disease_summary = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        print(f"\n{'=' * 80}")
        print(f'Disease Group Discovery : {dataset_name}')
        print(f"{'=' * 80}")

        if not matrix_files:

            print('No metadata files available.')


            disease_summary.append({
                'Dataset': dataset_name,
                'Disease Groups': 'Unknown',
                'Group Count': 0
            })

            continue

        discovered_groups = set()

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            for field, values in metadata.items():

                if 'characteristics' not in field.lower():
                    continue

                for value in values:

                    value = str(value).strip()

                    if ':' in value:

                        key, phenotype = value.split(':', 1)

                        key = key.strip().lower()
                        phenotype = phenotype.strip()

                        if any(
                            keyword in key
                            for keyword in [
                                'disease',
                                'group',
                                'phenotype',
                                'patient type',
                                'condition',
                                'status'
                            ]
                        ):
                            discovered_groups.add(phenotype)

        if discovered_groups:

            print('\nDisease groups identified:')
            tabulate(discovered_groups, headers='keys', tablefmt='grid')


        else:

            print(
                'No disease groups detected. '
                'Further inspection required.'
            )

        disease_summary.append({
            'Dataset': dataset_name,
            'Disease Groups':
                '; '.join(sorted(discovered_groups))
                if discovered_groups
                else 'Not identified',
            'Group Count': len(discovered_groups)
        })

    return pd.DataFrame(disease_summary)

## extracts sample titles and phenotype metadata
def extract_sample_annotations():
    annotations = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            accessions = metadata.get(
                '!Sample_geo_accession',
                []
            )

            titles = metadata.get(
                '!Sample_title',
                []
            )

            characteristics = metadata.get(
                '!Sample_characteristics_ch1',
                []
            )

            sample_count = min(
                len(accessions),
                len(titles)
            )

            for i in range(sample_count):
                annotations.append({
                    'Dataset': dataset_name,
                    'GSM': accessions[i],
                    'Title': titles[i],
                    'Characteristics':
                        characteristics[i]
                        if i < len(characteristics)
                        else 'N/A'
                })

    return pd.DataFrame(annotations)

## counts samples within each disease group
def summarise_group_distribution():

    from collections import Counter

    distribution = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        group_counter = Counter()

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            characteristics = metadata.get(
                '!Sample_characteristics_ch1',
                []
            )

            for item in characteristics:

                if ':' not in str(item):
                    continue

                key, value = item.split(':', 1)

                if 'patient type' in key.lower():

                    group_counter[value.strip()] += 1
        tabulate(group_counter.items(), headers=['Group', 'Count'], tablefmt='grid')

        for group, count in sorted(group_counter.items()):

            tabulate(group_counter.items(), headers=['Group', 'Count'], tablefmt='grid')

            distribution.append({
                'Dataset': dataset_name,
                'Group': group,
                'Samples': count
            })

    return pd.DataFrame(distribution)

## explores all sample characteristic categories
def explore_characteristic_dimensions():

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        print(f"\n{'=' * 80}")
        print(dataset_name)
        print(f"{'=' * 80}")

        dimensions = {}

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            characteristics = metadata.get(
                '!Sample_characteristics_ch1',
                []
            )

            for value in characteristics:

                if ':' not in str(value):
                    continue

                key, item = value.split(':', 1)

                key = key.strip()

                if key not in dimensions:
                    dimensions[key] = set()

                dimensions[key].add(item.strip())

        for dimension in sorted(dimensions):

            tabulate(dimensions[dimension], headers='keys', tablefmt='grid')

            for value in sorted(list(dimensions[dimension]))[:20]:
                print(f'  - {value}')

## counts unique GSM identifiers - GSM ids are unique sample identifiers in GEO datasets
def count_gsm_samples(folder):
    import re

    gsm_ids = set()

    for file in folder.rglob('*'):

        match = re.search(r'(GSM\d+)', file.name)

        if match:
            gsm_ids.add(match.group(1))

    return len(gsm_ids)


def discover_metadata_fields():
    field_summary = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)

            field_summary.append({
                'Dataset': dataset_name,
                'Metadata fields': len(metadata),
                'Fields': ', '.join(
                    sorted(metadata.keys())[:10]
                )
            })

    return pd.DataFrame(field_summary)


## summarizes available metadata fields
def summarise_metadata_fields():
    summary = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for matrix_file in matrix_files:
            metadata = extract_geo_metadata(matrix_file)

            summary.append({
                'Dataset': dataset_name,
                'Metadata fields': len(metadata),
                'Fields': ', '.join(
                    sorted(metadata.keys())[:10]
                )
            })

    return pd.DataFrame(summary)


## extracts phenotype information from GEO metadata - phenotype information is often stored in the 'characteristics' fields of GEO series matrix files
def extract_phenotypes():
    phenotype_rows = []

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            for field, values in metadata.items():

                if 'characteristics' in field.lower():
                    phenotype_rows.append({
                        'Dataset': dataset_name,
                        'Field': field,
                        'Values': '; '.join(
                            sorted(set(values[:20]))
                        )
                    })

    return pd.DataFrame(phenotype_rows)


## evaluates completeness of metadata
def assess_metadata_completeness():
    assessment = []

    required_fields = [
        '!Sample_title',
        '!Sample_geo_accession',
        '!Sample_source_name_ch1',
        '!Sample_characteristics_ch1'
    ]

    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['folder']

        matrix_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for matrix_file in matrix_files:

            metadata = extract_geo_metadata(matrix_file)

            missing = []

            for field in required_fields:

                if field not in metadata:
                    missing.append(field)

            assessment.append({
                'Dataset': dataset_name,
                'Missing fields': len(missing),
                'Status':
                    'COMPLETE'
                    if not missing
                    else 'INCOMPLETE'
            })

    return pd.DataFrame(assessment)



def inspect_expression_headers():
    for dataset_name, dataset_info in DATASETS.items():

        folder = dataset_info['resolved_folder']

        expression_files = list(
            folder.glob('*series_matrix*.txt.gz')
        )

        for expression_file in expression_files:

            print(f'\nInspecting expression headers in {expression_file.name}...')

            with gzip.open(expression_file, 'rt', errors='ignore') as f:

                for line in f:

                    if line.startswith('!'):
                        continue

                    headers = line.strip().split('\t')

                    tabulate(headers, headers='keys', tablefmt='grid')
                    break


## verifies sample metadata integrity
def assess_sample_integrity():
    sample_df = extract_sample_annotations()

    print("\nSample integrity report")

    print(
        tabulate(
            sample_df.head(20),
            headers='keys',
            tablefmt='grid'
        )
    )

    print(
        f'\nTotal samples discovered: '
        f'{sample_df["GSM"].nunique()}'
    )

    print(
        f'Missing GSM IDs: '
        f'{sample_df["GSM"].isna().sum()}'
    )

    print(
        f'Missing titles: '
        f'{sample_df["Title"].isna().sum()}'
    )

    print(
        f'Missing phenotype information: '
        f'{sample_df["Characteristics"].isna().sum()}'
    )
    return pd.DataFrame(sample_df)


def metadata_caller():
    inspect_metadata_files()
    discover_metadata_fields()
    assess_metadata_completeness()
    summarise_metadata_fields()

def data_existence_caller():
    validate_dataset_folders()
    inspect_dataset_files()

def biological_sample_caller():
    discover_biological_samples()
    discover_disease_groups()
    extract_sample_annotations()
    summarise_group_distribution()
    explore_characteristic_dimensions()

def data_artifact_caller():
    phenotype_df = extract_phenotypes()
    study_df = assess_sample_integrity()
    completeness_df = assess_metadata_completeness()

    tabulate(study_df, headers='keys', tablefmt='grid')
    tabulate(phenotype_df, headers='keys', tablefmt='grid')
    tabulate(completeness_df, headers='keys', tablefmt='grid')

    return study_df, phenotype_df, completeness_df

def export_dataframes_to_csv(study_df, phenotype_df, completeness_df):
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    study_df.to_csv(output_dir / 'study_data.csv', index=False)
    phenotype_df.to_csv(output_dir / 'phenotype_data.csv', index=False)
    completeness_df.to_csv(output_dir / 'metadata_completeness.csv', index=False)
def __main__():
    data_existence_caller()
    metadata_caller()
    biological_sample_caller()
    study_df, phenotype_df, completeness_df = data_artifact_caller()
    export_dataframes_to_csv(study_df, phenotype_df, completeness_df)





if __name__ == '__main__':
    __main__()
