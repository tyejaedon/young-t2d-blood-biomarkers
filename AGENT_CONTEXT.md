# AGENT_CONTEXT.md — AI Agent Guidance & Operating Rules

> **System Purpose:** This document serves as the master context and rulebook for any AI agent or assistant collaborating on Group E9's Bioinformatics Project. All generated code, documentation, and responses MUST align with the instructions herein.

---

## 1. Project Persona & Core Mandate

* **Project Title:** Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus
* **Group:** E9 | Track 6: Applied / Clinical Bioinformatics
* **Core Philosophy:** `Biological Question → Biological Data → Computational Analysis → Results → Biological Interpretation`
* **Primary Audience:** Computer Science students executing bioinformatics workflows. Always explain biological concepts in clear, accessible terms with defined jargon, while maintaining mathematical and statistical rigor.

---

## 2. Non-Negotiable Project Boundaries

1. **Organism & Tissue:** Strictly *Homo sapiens* human blood-derived transcriptomics (PBMCs or Whole Blood). Rejection rule: Never use solid tissue biopsies (e.g., liver, pancreatic islets) as the primary dataset.
2. **Age Demographic:** Target population is strictly young adults aged **18–40 years**. Rejection rule: Never include older adult cohorts (>40) in the primary statistical design without explicit filtering.
3. **Primary Datasets:**
   * **Primary Discovery:** NCBI GEO GSE26168 (Blood expression matrix).
   * **Independent Validation:** NCBI GEO GSE15932 (PBMC expression matrix).
   * **Cross-Tissue Context (Secondary Only):** GSE164416 (Pancreatic islets — for downstream pathway comparison only).
4. **Statistical Thresholds:**
   * $|\log_2\text{FC}| \ge 1.0$
   * Adjusted $p\text{-value} < 0.05$ (Benjamini-Hochberg FDR correction).

---

## 3. Mandatory Code & Engineering Standards

* **Headless Visualizations:** Always set `matplotlib.use('Agg')` before importing `pyplot` to prevent display server errors in headless terminal environments.
* **Directory Hygiene:** All scripts must write working outputs to `data/processed/`, figures to `results/figures/`, and tabular outputs to `results/tables/`.
* **Logging & Transparency:** Every Python script must print clear execution logs (matrix shapes, missing value counts, variance ratios).
* **Reproducibility:** Fix random seeds explicitly across stochastic operations (`np.random.seed(42)`, `random_state=42` in scikit-learn).

---

## 4. Common Failure Modes to Prevent

* **Failure Mode 1: Pure Machine Learning Focus.** Avoid presenting ML accuracy scores or code without explaining what the candidate genes mean biologically.
* **Failure Mode 2: Unnormalized Data Entry.** Always verify that input matrices are log2-transformed and quantile-normalized before running linear regression or feature selection.
* **Failure Mode 3: Overfitting on Single Dataset.** Always validate prioritized biomarkers on an independent secondary dataset (GSE15932).

---
