"""
Stage 4: Differential Expression Analysis (Discovery Cohort GSE26168)
Orchestrates:
- Limma Empirical Bayes statistical modeling (T2D vs Control)
- Dual low-expression filtering comparative analysis (Approach 1 vs Approach 2)
- Monotonic early-dysregulation trajectory evaluation across glycemic stages
- Export of diagnostic tables to results/tables/ and publication figures to results/figures/
"""

import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing_config import DEFAULT_PREPROCESSING_CONFIG
from dea.discovery_analysis import run_discovery_dea_pipeline


def main():
    results = run_discovery_dea_pipeline(DEFAULT_PREPROCESSING_CONFIG)
    print("\n[SUMMARY] Differential Expression Analysis Execution Summary:")
    print(f"  - Total Features Tested: {results['metrics']['total_tested']:,}")
    print(f"  - Statistically Significant DEGs (|log2FC| >= 1.0, FDR < 0.05): {results['metrics']['sig_degs']}")
    print(f"    * Up-regulated: {results['metrics']['up_regulated']}")
    print(f"    * Down-regulated: {results['metrics']['down_regulated']}")
    print(f"  - Concordant Core Genes Across Both Filtering Approaches: {results['metrics']['shared_core_genes']}")
    print("\nGenerated Tables:")
    for name, path in results["tables"].items():
        print(f"  - {name}: {path}")
    print("\nGenerated Figures:")
    for name, path in results["figures"].items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
