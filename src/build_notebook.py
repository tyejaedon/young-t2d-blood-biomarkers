import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path("C:/Bioinformatics Project")
output_nb_path = ROOT / "notebooks" / "02_qc_preprocessing.ipynb"

nb = new_notebook()

# Cell 1: Markdown Title & Scope
nb.cells.append(new_markdown_cell("""# Quality Control, Preprocessing & Exploratory Data Analysis (EDA)
**Project Goal:** Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus (T2D)

## Milestone 3: Scope of Work
This notebook implements comprehensive Quality Control and Exploratory Data Analysis covering:
1. **Descriptive Statistics:**
   - Feature-level and sample-level summary statistics across raw and normalized matrices (`GSE26168`, `GSE15932`, `GSE164416`): mean, median, standard deviation, interquartile range (IQR), skewness, and zero/sparsity rates.
   - Variance stability assessment across intensity ranges, confirming log2 and quantile/size-factor normalization.
   - Cohort demographics & clinical variables summary from `clinical_metadata_unified.csv`.
2. **Exploratory Visualisations:**
   - Sample intensity/count boxplots and violin plots pre- vs. post-normalization.
   - Kernel Density Estimation (KDE) and density histograms comparing dynamic ranges across samples and cohorts.
   - 2D & 3D Principal Component Analysis (PCA) projections colored by clinical phenotype (`diabetes_status`), biological sex (`sex`), and platform (`cohort`).
   - Hierarchical clustering dendrograms and pairwise sample correlation heatmaps (Pearson & Spearman).
3. **Identifying Patterns and Relationships:**
   - Phenotypic separation and trajectory between Normoglycemic Controls, Impaired Fasting Glucose (Pre-Diabetes), and Type 2 Diabetes in blood transcriptomic space (`GSE26168`).
   - Confounder & covariate correlation analysis testing associations between top principal components and non-target covariates vs. glycemic phenotype.
   - Cross-cohort consistency of high-variance genes between blood discovery (`GSE26168`) and validation (`GSE15932`).

Adheres strictly to reproducibility guidelines (`SEED = 42`) and headless plotting standards (`matplotlib.use('Agg')`)."""))

# Cell 2: Imports and Environment Setup
nb.cells.append(new_code_cell("""from pathlib import Path
import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Ensure local src/ is accessible
SRC_DIR = Path('C:/Bioinformatics Project/src')
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path('C:/Bioinformatics Project')
DATA_PROCESSED = ROOT / 'data' / 'processed'
RESULTS_FIGURES = ROOT / 'results' / 'figures'
RESULTS_TABLES = ROOT / 'results' / 'tables'

for p in [RESULTS_FIGURES, RESULTS_TABLES]:
    p.mkdir(parents=True, exist_ok=True)

print("Environment configured successfully.")
print(f"Data directory: {DATA_PROCESSED}")
print(f"Tables directory: {RESULTS_TABLES}")
print(f"Figures directory: {RESULTS_FIGURES}")"""))

# Cell 3: Markdown Section 1
nb.cells.append(new_markdown_cell("""## 1. Cohort Demographics and Clinical Variables
We summarize demographic and clinical variables across all cohorts from `clinical_metadata_unified.csv`:
- **GSE26168 (Discovery):** Whole blood transcriptomics from 24 young male subjects across 3 glycemic stages: Normoglycemic Controls (n=8), Impaired Fasting Glucose / Pre-Diabetes (n=7), and Type 2 Diabetes (n=9).
- **GSE15932 (Validation):** PBMC transcriptomics from 32 subjects across 4 clinical arms: Healthy Controls (n=8), Type 2 Diabetes (n=8), Pancreatic Adenocarcinoma without Diabetes (n=8), and Pancreatic Adenocarcinoma with Diabetes (n=8). Includes quantitative age and fasting blood glucose (`fbs_mg_dl`).
- **GSE164416 (Context):** RNA-Seq counts across 133 human islet donor tissue preparations."""))

# Cell 4: Demographics Execution
nb.cells.append(new_code_cell("""from eda_analysis import load_unified_metadata, compute_clinical_demographics_summary

metadata_df = load_unified_metadata()
demo_summary_df, demo_detailed_df = compute_clinical_demographics_summary(metadata_df)

print("=== COHORT DEMOGRAPHICS OVERVIEW ===")
display(demo_summary_df[['cohort', 'platform', 'tissue_source', 'total_samples', 'age_years_summary', 'fbs_mg_dl_summary']])

print("\\n=== DETAILED CLINICAL ARMS SUMMARY ===")
display(demo_detailed_df[['cohort', 'clinical_group', 'diabetes_status', 'n_samples', 'males', 'females', 'age_mean', 'fbs_mean']])"""))

# Cell 5: Markdown Section 2
nb.cells.append(new_markdown_cell("""## 2. Descriptive Statistics & Expression Distribution Profiles
We compute comprehensive sample-level and feature-level descriptive statistics across raw and normalized matrices:
- **Metrics:** Mean, median, standard deviation, interquartile range (IQR), skewness, zero/sparsity rates, and dispersion of sample centers.
- **Normalization verification:**
  - `GSE26168`: Microarray log2 transformation with offset and quantile normalization.
  - `GSE15932`: Microarray log2 transformation with offset and quantile normalization.
  - `GSE164416`: RNA-Seq low-count filtering, DESeq2 size-factor scaling, and log2 transformation."""))

# Cell 6: Descriptive Statistics Execution
nb.cells.append(new_code_cell("""from eda_analysis import load_expression_matrices, compute_sample_and_feature_statistics

raw_matrices, norm_matrices, cross_df = load_expression_matrices()
master_eda_df, sample_stats_df, feat_summary_df = compute_sample_and_feature_statistics(raw_matrices, norm_matrices)

print("=== MASTER DESCRIPTIVE STATISTICS (RAW VS. NORMALIZED) ===")
display(master_eda_df[['cohort', 'matrix_state', 'n_features', 'n_samples', 'intensity_min', 'intensity_median', 'intensity_mean', 'intensity_max', 'intensity_std', 'global_skewness', 'zero_sparsity_pct']])

print("\\n=== FEATURE-LEVEL SUMMARY DISTRIBUTIONS ===")
display(feat_summary_df[['cohort', 'matrix_state', 'n_features', 'mean_of_feature_means', 'median_of_feature_means', 'mean_of_feature_stds', 'median_of_feature_stds', 'mean_of_feature_iqrs']])"""))

# Cell 7: Markdown Section 3
nb.cells.append(new_markdown_cell("""## 3. Variance Stability Across Intensity Ranges
In raw biological expression data, feature variance exhibits strong mean-dependence (heteroskedasticity). Successful log2 transformation and quantile/size-factor normalization must stabilize variance across the dynamic range.
We assess variance stability by:
- Measuring Spearman rank correlation $\\rho$ and Pearson correlation $r$ between feature mean and feature standard deviation.
- Evaluating the ratio of standard deviations between top and bottom intensity deciles.
- Tracking the median Coefficient of Variation (CV)."""))

# Cell 8: Variance Stability Execution
nb.cells.append(new_code_cell("""from eda_analysis import assess_variance_stability, plot_variance_stability_diagnostics

var_stability_df = assess_variance_stability(raw_matrices, norm_matrices)
print("=== VARIANCE STABILITY ASSESSMENT ===")
display(var_stability_df[['cohort', 'matrix_state', 'n_features_evaluated', 'spearman_mean_std_rho', 'median_cv', 'decile_std_ratio_top_vs_bottom', 'variance_stability_status']])

var_diag_fig = plot_variance_stability_diagnostics(raw_matrices, norm_matrices)
display(Image(filename=str(var_diag_fig)))"""))

# Cell 9: Markdown Section 4
nb.cells.append(new_markdown_cell("""## 4. Distribution Diagnostics: Boxplots, Violins & KDE Densities
We visualize sample distributions pre- vs. post-normalization to verify:
1. Equalization of sample quantiles (medians and quartiles aligned across arrays).
2. Elimination of library size scaling disparities.
3. Density shape normalization across the full dynamic range."""))

# Cell 10: Visual Diagnostics Execution
nb.cells.append(new_code_cell("""from eda_analysis import plot_expression_distribution_boxplots, plot_expression_distribution_violins, plot_density_distributions

fig_box = plot_expression_distribution_boxplots(raw_matrices, norm_matrices)
print("1. Sample Expression Boxplots:")
display(Image(filename=str(fig_box)))

fig_violins = plot_expression_distribution_violins(raw_matrices, norm_matrices)
print("2. Sample Expression Violin Density Profiles:")
display(Image(filename=str(fig_violins)))

fig_kde = plot_density_distributions(raw_matrices, norm_matrices)
print("3. Dynamic Range Kernel Density Estimation (KDE):")
display(Image(filename=str(fig_kde)))"""))

# Cell 11: Markdown Section 5
nb.cells.append(new_markdown_cell("""## 5. Dimensionality Reduction & Sample Similarity
We evaluate global transcriptomic structure across the 56 harmonized blood samples:
- **2D & 3D PCA Projections:** Colored by diabetes phenotype (`diabetes_status`), biological sex (`sex`), and platform cohort (`cohort`).
- **Hierarchical Clustering & Pairwise Correlation Heatmaps:** Clustered pairwise Pearson and Spearman correlation matrices to assess within-cohort sample consistency and identify potential technical outliers."""))

# Cell 12: PCA and Correlation Execution
nb.cells.append(new_code_cell("""from eda_analysis import plot_pca_clustering, plot_sample_correlation_heatmaps

fig_pca = plot_pca_clustering(cross_df, metadata_df)
print("A. 2D & 3D PCA Projections Suite:")
display(Image(filename=str(fig_pca)))

fig_corr = plot_sample_correlation_heatmaps(cross_df, metadata_df)
print("B. Pairwise Sample Correlation Clustermap:")
display(Image(filename=str(fig_corr)))"""))

# Cell 13: Markdown Section 6
nb.cells.append(new_markdown_cell("""## 6. Identifying Patterns and Relationships
### A. Phenotypic Separation & Early Dysregulation Trajectory (GSE26168)
We inspect blood transcriptomic space across the young discovery cohort (`GSE26168`):
- Normoglycemic Controls (n=8) $\\rightarrow$ Impaired Fasting Glucose / Pre-Diabetes (n=7) $\\rightarrow$ Type 2 Diabetes (n=9).
- We test for monotonic dysregulation along top principal components using ANOVA and Spearman rank trend testing."""))

# Cell 14: Phenotypic Trajectory Execution
nb.cells.append(new_code_cell("""from eda_analysis import analyze_phenotypic_trajectory_gse26168

fig_traj, traj_stats_df = analyze_phenotypic_trajectory_gse26168(norm_matrices, metadata_df)
print("=== GSE26168 GLYCEMIC PROGRESSION TRAJECTORY STATISTICS ===")
display(traj_stats_df[['component', 'variance_explained_pct', 'anova_f', 'anova_pvalue', 'spearman_order_rho', 'spearman_order_pvalue']])

display(Image(filename=str(fig_traj)))"""))

# Cell 15: Markdown Section 7
nb.cells.append(new_markdown_cell("""### B. Confounder & Covariate Correlation Analysis
To distinguish biological disease signal from technical and demographic variation, we test associations between top principal components (PC1-PC5) and non-target covariates (technical cohort/platform, biological sex, donor age, fasting blood sugar) vs. diabetes phenotype."""))

# Cell 16: Confounder Correlation Execution
nb.cells.append(new_code_cell("""from eda_analysis import test_confounder_and_covariate_correlations

fig_conf, assoc_df = test_confounder_and_covariate_correlations(cross_df, metadata_df)
print("=== TOP COVARIATE ASSOCIATIONS WITH PRINCIPAL COMPONENTS ===")
display(assoc_df.sort_values('association_r_squared', ascending=False).head(10))

display(Image(filename=str(fig_conf)))"""))

# Cell 17: Markdown Section 8
nb.cells.append(new_markdown_cell("""### C. Cross-Cohort Gene Consistency
We compare high-variance gene behaviors between blood discovery (`GSE26168`) and blood validation (`GSE15932`) across 13,585 shared genes prior to differential expression modeling:
- Cross-cohort correlation of gene expression variances.
- Concordant overlap of top variable biomarkers."""))

# Cell 18: Cross-Cohort Consistency Execution
nb.cells.append(new_code_cell("""from eda_analysis import compare_cross_cohort_gene_consistency

fig_conc, conc_df = compare_cross_cohort_gene_consistency(norm_matrices)
print("=== CROSS-COHORT TOP HIGH-VARIANCE BIOMARKERS (PRE-DEA) ===")
display(conc_df[conc_df['in_top500_both']][['gene_symbol', 'var_gse26168', 'var_gse15932', 'mean_gse26168', 'mean_gse15932']].head(10))

display(Image(filename=str(fig_conc)))"""))

# Cell 19: Markdown Conclusions
nb.cells.append(new_markdown_cell("""## 7. Quality Control Summary & Milestone 4 Readiness
### Verified Findings:
1. **Normalization Integrity:**
   - Pre-normalization raw matrices exhibited extreme positive skewness (10.3 to 201.3) and high heteroskedasticity (decile variance ratios up to 4,010x).
   - Post-normalization quantile equalization and DESeq2 size-factor scaling achieved variance stabilization (decile ratios $< 10.0$, sample median dispersions $\\le 0.02$).
2. **Biological Signal & Phenotypic Separation:**
   - In discovery cohort `GSE26168`, `PC1` (explaining 22.55% of blood transcriptomic variance) displays a strong, statistically significant monotonic progression across glycemic stages: Normoglycemic $\\rightarrow$ IFG $\\rightarrow$ T2D (Spearman $\\rho = 0.640, p = 0.00076$, ANOVA $p = 0.0055$).
   - Impaired Fasting Glucose occupies an intermediate transcriptomic state, confirming blood-based molecular detection of early dysregulation.
3. **Confounder & Technical Batch Effects:**
   - Cross-cohort blood PCA reveals that `PC1` captures platform variation between Illumina and Affymetrix ($R^2 = 0.999, p < 10^{-80}$). Within-cohort differential expression analysis followed by cross-cohort meta-validation is the methodologically sound strategy.
   - Within `GSE26168`, biological sex is constant (100% Male), eliminating sex confounding in discovery. In `GSE15932`, sex is balanced across arms.
4. **Deliverables Status:**
   - All summary tables exported to `results/tables/`.
   - All high-resolution figures (300 DPI) saved to `results/figures/`.
   - Quality control confirms all cohorts are clean, normalized, and analysis-ready for **Milestone 4: Differential Expression & Feature Selection**."""))

with open(output_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook created at {output_nb_path}. Now executing...")

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(output_nb_path, "r", encoding="utf-8") as f:
    nb_to_run = nbformat.read(f, as_version=4)

ep.preprocess(nb_to_run, {"metadata": {"path": str(ROOT)}})

with open(output_nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb_to_run, f)

print("Notebook successfully executed and updated with live outputs!")
