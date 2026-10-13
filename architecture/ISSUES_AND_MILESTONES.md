# Issues and Milestones Plan

This document breaks the project goal — **identifying gene expression biomarkers for early detection of young-onset Type 2 Diabetes Mellitus (T2D)** — into milestones and actionable issues aligned with the repository blueprint.

## Milestone 1 — Problem Definition & Scope
**Goal:** Lock the biological question, inclusion criteria, and success metrics.

### Issues
- **I-01: Confirm research question and hypothesis**
  - Define the exact comparison: young adults (18–40) with T2D vs. normoglycemic controls.
  - **Acceptance criteria:** One-sentence research question and hypothesis recorded in `architecture` or notebook setup.
- **I-02: Finalize dataset selection policy**
  - Confirm primary, validation, and context datasets.
  - **Acceptance criteria:** `GSE26168` = discovery, `GSE15932` = validation, `GSE164416` = secondary context only.
- **I-03: Define analysis success metrics**
  - Set statistical thresholds and ML evaluation metrics.
  - **Acceptance criteria:** Thresholds documented (`|log2FC| ≥ 1`, FDR < 0.05, AUC/sensitivity/specificity reported).
- **I-04: Create notebook and folder scaffolding** — **Complete**
  - Ensure the notebook sequence and output directories exist.
  - **Acceptance criteria:** `../notebooks/00_project_setup.ipynb` through `05_biomarker_modeling.ipynb` are present; `../dataset/processed`, `results/figures/`, and `results/tables/` are used consistently.

**Milestone deliverable:** Approved project scope and reproducible workspace structure.

---

## Milestone 2 — Data Acquisition & Inventory
**Goal:** Retrieve datasets and document metadata, sample counts, and cohort structure.

### Issues
- **I-05: Download and organize GEO discovery dataset**
  - Acquire `GSE26168` and place raw files in the correct location.
  - **Acceptance criteria:** Raw discovery data is present and traceable.
- **I-06: Download and organize GEO validation dataset**
  - Acquire `GSE15932` and place raw files in the correct location.
  - **Acceptance criteria:** Validation dataset is present and traceable.
- **I-07: Create a dataset inventory table**
  - Record accession, platform, tissue, sample counts, phenotype labels, and notes.
  - **Acceptance criteria:** Inventory table exists in `../notebooks/01_data_inventory.ipynb` or `results/tables/`.
- **I-08: Verify demographic and tissue constraints**
  - Ensure the primary analysis remains within human blood/PBMC samples and the target age range.
  - **Acceptance criteria:** A short data-governance note confirms compliance with project boundaries.

**Milestone deliverable:** Dataset inventory with justified selection of primary and validation cohorts.

---

## Milestone 3 — Data Preparation & Quality Control
**Goal:** Clean, transform, normalize, and inspect expression matrices before downstream analysis.

### Issues
- **I-09: Build data import and parsing pipeline**
  - Load expression matrices and metadata into a consistent format.
  - **Acceptance criteria:** Input shapes and metadata columns are logged.
- **I-10: Implement missing-value and consistency checks**
  - Detect missing values, duplicate sample IDs, and malformed annotations.
  - **Acceptance criteria:** QC summary reports missingness and sample integrity.
- **I-11: Apply log2 transformation and normalization**
  - Apply the required transformation and quantile normalization where appropriate.
  - **Acceptance criteria:** Normalized matrix saved to `../dataset/processed`.
- **I-12: Filter low-expression genes**
  - Remove genes below the chosen background threshold.
  - **Acceptance criteria:** Filtering rule is documented and number of retained features is reported.
- **I-13: Produce QC visualizations**
  - Create boxplots, PCA plots, and any needed sample-level diagnostics.
  - **Acceptance criteria:** Figures are saved to `../results/figures` and support the QC conclusion.
- **I-38: Exploratory Data Analysis (EDA): Descriptive statistics, exploratory visualisations, and pattern identification** — **Complete**
  - Compute descriptive statistics across cohorts, generate distribution/sample correlation/PCA visualisations, and assess phenotypic pattern separation and confounders.
  - **Acceptance criteria:** Tables saved to `results/tables/`, high-resolution headless figures saved to `results/figures/`, and pattern insights documented in `notebooks/02_qc_preprocessing.ipynb`.

**Milestone deliverable:** Clean, normalized matrices with QC evidence that samples are usable.

---

## Milestone 4 — Differential Expression & Feature Selection
**Goal:** Identify candidate genes associated with T2D and reduce them to a robust biomarker shortlist.

### Issues
- **I-14: Run differential expression analysis on discovery dataset**
  - Use an appropriate method for the assay type.
  - **Acceptance criteria:** DEG table generated with `|log2FC| ≥ 1` and FDR < 0.05.
- **I-15: Validate directionality and effect sizes**
  - Confirm upregulated and downregulated genes are clearly separated.
  - **Acceptance criteria:** Volcano plot and ranked DEG table are produced.
- **I-16: Perform feature reduction / selection**
  - Apply LASSO, RFE, or a comparable feature-selection method.
  - **Acceptance criteria:** A compact candidate-gene shortlist is produced.
- **I-17: Train and compare classification models**
  - Evaluate Random Forest, SVM, and Logistic Regression.
  - **Acceptance criteria:** Cross-validation metrics are reported for each model.
- **I-18: Assess validation performance on the independent dataset**
  - Test shortlisted biomarkers on `GSE15932`.
  - **Acceptance criteria:** Validation results are documented and compared against discovery findings.

**Milestone deliverable:** Ranked biomarker candidates with differential expression and model performance outputs.

---

## Milestone 5 — Pathway Enrichment & Biological Interpretation
**Goal:** Convert statistical findings into a biological narrative.

### Issues
- **I-19: Run functional enrichment analysis**
  - Analyze GO, KEGG, and Reactome enrichment.
  - **Acceptance criteria:** Enrichment results table is produced and interpretable.
- **I-20: Interpret top genes in the T2D context**
  - Link candidate genes to beta-cell stress, immune activation, ER stress, and metabolic dysfunction.
  - **Acceptance criteria:** Written interpretation explains how each major gene/pathway fits the disease model.
- **I-21: Compare blood findings with cross-tissue context data**
  - Use `GSE164416` only as secondary biological context.
  - **Acceptance criteria:** A short comparative note explains any overlapping or contrasting signals.
- **I-22: Document limitations and future validation steps**
  - Record sample-size constraints, retrospective bias, and wet-lab follow-up ideas.
  - **Acceptance criteria:** Limitations section is ready for the final report.

**Milestone deliverable:** Biological interpretation draft grounded in human T2D pathophysiology.

---

## Milestone 6 — Final Presentation & Documentation
**Goal:** Package the project into a polished, reproducible final submission.

### Issues
- **I-23: Consolidate notebook outputs**
  - Ensure all notebooks run in sequence and outputs land in the correct folders.
  - **Acceptance criteria:** Notebook chain is coherent and reproducible.
- **I-24: Prepare final tables and figures**
  - Curate the best plots, DEG tables, and summary visuals.
  - **Acceptance criteria:** Final assets are organized under `../results/figures` and `results/tables/`.
- **I-25: Write the final report**
  - Summarize question, methods, results, and interpretation.
  - **Acceptance criteria:** Report covers all checkpoint requirements and biological conclusions.
- **I-26: Prepare the presentation deck**
  - Convert the report into a concise slide deck.
  - **Acceptance criteria:** Slides clearly state the problem, workflow, findings, and significance.
- **I-27: Final quality review**
  - Check consistency of terminology, citations, filenames, and outputs.
  - **Acceptance criteria:** No missing deliverables or broken references remain.

**Milestone deliverable:** Complete final submission package ready for grading or review.

---

## Suggested Milestone Order
1. Problem Definition & Scope
2. Data Acquisition & Inventory
3. Data Preparation & Quality Control
4. Differential Expression & Feature Selection
5. Pathway Enrichment & Biological Interpretation
6. Final Presentation & Documentation

---

## Notes
- The plan follows the project blueprint and checkpoint structure in `architecture`.
- Issues are intentionally ordered to support a strict biological-first workflow.
- If you want, this document can be split into separate GitHub-style issue templates or converted into a kanban-ready checklist.
