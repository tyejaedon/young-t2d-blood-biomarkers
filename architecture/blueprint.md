# Project Blueprint: Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus

**Course Track:** Track 6: Applied / Clinical Bioinformatics  
**Group:** E9  
**Core Framework:** `Biological Question → Biological Data → Computational Analysis → Results → Biological Interpretation`

---

## 1. Executive Summary & Core Principle

This blueprint defines the architecture, workflow, and quality gates for Group E9's bioinformatics research project. The project addresses a critical diagnostic challenge: **young-onset Type 2 Diabetes Mellitus (T2D) in adults aged 18–40**, which exhibits an aggressive disease trajectory but frequently escapes early clinical detection due to the limitations of traditional glycemic indicators ($\text{HbA1c}$, fasting blood glucose).

By analyzing publicly available blood-derived transcriptomic profiles (Microarray and RNA-seq), this project aims to identify early gene expression shifts and prioritize non-invasive candidate biomarkers. Every phase of development strictly adheres to the core bioinformatics directive: **no algorithm or code is executed without explicit biological justification and clinical interpretation.**

---

## 2. Research Architecture & Scope

| Dimension | Specification | Strategic Rationale |
| :--- | :--- | :--- |
| **Organism** | *Homo sapiens* (Human) | Ensures translational relevance to human pathophysiology. |
| **Target Population** | Young Adults aged **18–40 years** | Disengages young-onset T2D mechanisms from age-related systemic decline. |
| **Tissue Modality** | Peripheral Blood Mononuclear Cells (PBMCs) / Whole Blood | Targets non-invasive screening accessible via standard venipuncture. |
| **Phenotype Comparison** | T2D Cases vs. Normoglycemic Healthy Controls | Enables unconfounded differential gene expression screening. |
| **Primary Accessions** | **GSE26168** (Primary Discovery) / **GSE15932** (Validation) | High-dimensional expression profiles with documented clinical metadata. |
| **Key Stakeholders** | Translational researchers, clinical diagnostic developers, endocrinologists | Actionable molecular targets for early screening assays. |

---

## 3. Stage-by-Stage Implementation Plan

### Stage 1: Problem Definition & Scope (Checkpoint 1)
* **Goal:** Establish biological rationale and frame research question.
* **Central Question:** *Can computational analysis of blood-derived transcriptomic profiles distinguish young adults (ages 18–40) with T2D from healthy controls and yield predictive candidate biomarkers?*
* **Specific Aims:**
  1. *Aim 1:* Dataset Acquisition & Preprocessing.
  2. *Aim 2:* Differential Expression Analysis (DEA).
  3. *Aim 3:* Functional Annotation & Pathway Enrichment.
  4. *Aim 4:* Biomarker Prioritization & Classification Modeling.

### Stage 2: Data Identification & Retrieval (Checkpoint 2)
* **Goal:** Retrieve and evaluate public GEO datasets adhering to MIAME/MINSEQE standards.
* **Primary Discovery Dataset:** `GSE26168` (Illumina / Affymetrix blood microarray).
* **Secondary Validation Dataset:** `GSE15932` (Affymetrix PBMC array).
* **Cross-Tissue Context Dataset:** `GSE164416` (Pancreatic islets - for secondary pathway contextualization).

### Stage 3: Data Preparation & Quality Control (Checkpoint 3)
* **Goal:** Sanitize, log-transform, and normalize high-dimensional matrices.
* **Preprocessing Pipeline:**
  * Ingestion & missing value check (`GEOparse` / `pandas`).
  * $\log_2(x + 1)$ transformation and Quantile Normalization.
  * **Low-Expression Filtering Strategy (Dual Comparative Analysis):**
    * *Approach 1 (Heuristic / Absolute Floor):* Total count sum $\ge 10$ & $\ge 5$ expressing samples for RNA-seq; $\le 80\%$ background floor for microarrays.
    * *Approach 2 (Distribution-Aware / Cohort-Fraction):* Group-aware CPM ($\text{CPM} \ge 1.0$ across $\ge \min(n_{\text{group}})$ samples) with DESeq2 independent filtering; non-specific variance/IQR filtering with MaxMean probe collapse for microarrays.
    * Track both approaches head-to-head across all downstream biological outcomes (DEA, pathway enrichment, and ML classification).
  * Outlier detection via sample intensity Boxplots and Principal Component Analysis (PCA).

### Stage 4: Differential Expression & Machine Learning (Checkpoint 4)
* **Goal:** Identify dysregulated transcripts and train predictive classifiers.
* **Statistical DEA:** Apply `limma` (microarray) or `DESeq2` (RNA-seq) with strict thresholds:
  * $|\log_2\text{FC}| \ge 1.0$
  * Adjusted $p \text{-value} < 0.05$ (Benjamini-Hochberg FDR correction).
* **Feature Selection & ML Modeling:**
  * Apply LASSO / Recursive Feature Elimination (RFE) to narrow top candidate genes.
  * Train supervised classifiers (Random Forest, SVM, Logistic Regression).
  * Evaluate cross-validation AUC-ROC, sensitivity, and specificity.
* **Functional Pathway Analysis:**
  * Gene Ontology (GO: Biological Process, Cellular Component, Molecular Function).
  * KEGG and Reactome over-representation analysis via `clusterProfiler` / `Enrichr`.

### Stage 5: Biological Interpretation & Synthesis (Checkpoint 5)
* **Goal:** Contextualize findings within human metabolic and immune biology.
* **Mechanistic Deep Dive:** Link top DEGs and pathways back to cellular stress, $\beta$-cell burnout, ER stress, and PBMC immune activation.
* **Limitations & Future Work:** Document sample size constraints, retrospective bias, and propose downstream wet-lab validation (e.g., qRT-PCR).

### Stage 6: Final Presentation & Documentation
* **Goal:** Synthesize the complete mini-research study into a final presentation and final report satisfying all rubric dimensions.

---

## 4. Master Rubric & Checkpoint Quality Matrix

| Checkpoint | Target Deliverable | Rubric Success Criteria |
| :--- | :--- | :--- |
| **CP 1** | Problem Statement & Scope | Clear biological context; focused 18–40 age bracket; 4 defined aims. |
| **CP 2** | Data Acquisition Report | Justified repository selection (GEO); non-invasive blood modality confirmed. |
| **CP 3** | Preprocessing & QC Artifacts | Demonstrated log2 transformation, quantile normalization, and clean PCA plots. |
| **CP 4** | DEA & ML Results | Correct statistical parameters applied ($|\log_2\text{FC}| \ge 1$, adj $p<0.05$); ML model metrics reported. |
| **CP 5** | Biological Interpretation | Thorough biological explanation connecting DEGs to T2D pathomechanisms. |

---
