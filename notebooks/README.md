# Notebook Workflow

This project is developed primarily in Jupyter notebooks. Use the numbered notebooks in order:

1. `00_project_setup.ipynb` - Environment verification, directory structure creation, and dependency verification.
2. `01_data_inventory.ipynb` - Cohort download status, integrity checks, and clinical metadata harmonization.
3. `02_qc_preprocessing.ipynb` - QC distributions, boxplots, variance stabilization, and PCA across raw vs. normalized matrices.
4. `03_differential_expression.ipynb` - Case vs. control design matrix construction, `limma`/`DESeq2` modeling, and DEG identification ($|\log_2\text{FC}| \ge 1.0$, FDR-adjusted $p < 0.05$).
5. `04_pathway_enrichment.ipynb` - Gene Ontology (GO) and KEGG/Reactome pathway over-representation analysis.
6. `05_biomarker_modeling.ipynb` - Machine learning feature selection (LASSO/RFE), classifier training, and external validation on `GSE15932`.

---

## Dual Low-Expression Filtering & Comparative Biological Analysis

As documented in the root [`README.md`](file:///C:/Bioinformatics%20Project/README.md) and [`src/preprocessing/normalization.py`](file:///C:/Bioinformatics%20Project/src/preprocessing/normalization.py), the notebooks execute and evaluate **two complementary low-expression filtering approaches**:
- **Approach 1 (Heuristic / Absolute Floor):** Total count sum $\ge 10$ and minimum expressing sample count for RNA-seq ([`filter_low_counts`](file:///C:/Bioinformatics%20Project/src/preprocessing/normalization.py#L273-L298)); background floor thresholding ($\le 80\%$) for microarrays.
- **Approach 2 (Distribution-Aware / Cohort-Fraction):** Group-aware CPM ($\text{CPM} \ge 1.0$ across $\ge \min(n_{\text{group}})$ samples) with DESeq2 independent filtering; non-specific variance / IQR filtering with MaxMean probe selection for microarrays.

The notebooks systematically track both filtering approaches across the full pipeline:
- **Notebook 02:** Compares retained transcript counts, zero-sparsity rates, and PCA clustering.
- **Notebook 03:** Compares statistical power, multiple testing burden (FDR), and DEG yield.
- **Notebook 04:** Compares biological pathway coverage and enrichment significance for immune/insulin signaling networks.
- **Notebook 05:** Compares classification diagnostic performance (AUC-ROC, sensitivity, specificity) and gene panel stability across cross-validation folds.
