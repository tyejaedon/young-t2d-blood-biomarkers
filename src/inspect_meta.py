import gzip
from pathlib import Path
from collections import defaultdict

def parse_full_meta(path):
    print("=" * 60)
    print("FILE:", path.name)
    print("=" * 60)
    meta = defaultdict(list)
    with gzip.open(path, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_'):
                parts = [p.replace('"', '').strip() for p in line.strip().split('\t')]
                key = parts[0]
                meta[key].append(parts[1:])
            elif line.startswith('!series_matrix_table_begin'):
                break

    accessions = meta['!Sample_geo_accession'][0]
    titles = meta['!Sample_title'][0] if '!Sample_title' in meta else []
    print(f"Total Samples: {len(accessions)}")
    print("First 3 GSMs:", accessions[:3])
    print("First 3 Titles:", titles[:3])

    if '!Sample_characteristics_ch1' in meta:
        for i, char_row in enumerate(meta['!Sample_characteristics_ch1']):
            key_vals = [c.split(':', 1) for c in char_row if ':' in c]
            if key_vals:
                k = key_vals[0][0].strip()
                vals = [kv[1].strip() for kv in key_vals]
                unique_vals = sorted(list(set(vals)))
                print(f"Row {i} [{k}] -> {len(unique_vals)} unique values: {unique_vals[:8]}")

if __name__ == '__main__':
    parse_full_meta(Path(r"C:/Bioinformatics Project/dataset/raw/GSE26168_RAW/GSE26168-GPL6883_series_matrix.txt.gz"))
    parse_full_meta(Path(r"C:/Bioinformatics Project/dataset/raw/GSE26168_RAW/GSE26168-GPL10322_series_matrix.txt.gz"))
    parse_full_meta(Path(r"C:/Bioinformatics Project/dataset/raw/GSE15932_RAW/GSE15932_series_matrix.txt.gz"))
