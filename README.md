# Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus

## Project Overview

This repository contains the complete computational workflow, notebook-based analysis environment, statistical models, and documentation for Group E9's research project: **"Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus"**.

Standard clinical screening tools (HbA1c, fasting blood glucose) flag diabetes only after substantial, irreversible pancreatic beta-cell exhaustion has taken root. This study leverages public blood-derived transcriptomic profiles (Microarray / RNA-seq) from NCBI GEO to identify early molecular markers of systemic cellular stress in young adults (ages 18–40).
---
## Research Objectives & Questions
The overarching question this project aspires to uncover is, can computational analysis of public blood-derived transcriptomic profiles distinguish young adults (ages 18–40) with Type 2 Diabetes Mellitus (T2D) from healthy controls and identify high-confidence candidate biomarkers for early detection?

 **To answer this research question, the project executes four sequential aims:** 
1. Dataset Curation & Quality Control: Retrieve human blood transcriptomic datasets (Microarray or RNA-seq) from NCBI GEO/ArrayExpress, filter for young adult cohorts (ages 18–40), and perform standard quality checks, missing-value handling, and cross-sample normalization. 
2. Differential Expression Analysis (DEA): Identify significantly dysregulated genes between T2D cases and normoglycemic controls using statistical methods (limma or DESeq2) with strict filtering thresholds (|log₂FC| ≥ 1 and adjusted p<0.05). 
3. Functional Pathway Annotation: Map differentially expressed genes to biological processes via Gene Ontology (GO) and KEGG/Reactome pathway over-representation analysis to uncover early metabolic and inflammatory dysregulations.
4. Biomarker Selection & Classification: Apply feature selection algorithms to narrow down top gene candidates and train a supervised classifier (e.g., Random Forest or Support Vector Machine) to evaluate biomarker accuracy and predictive performance.

---
## Hypothesis
We hypothesize that young adults with early-onset T2D exhibit a distinct blood transcriptomic signature compared to healthy controls, and that a subset of these differentially expressed genes can serve as reliable biomarkers for early detection, potentially enabling preemptive clinical interventions.
### Central Primary Hypothesis (H1):
> Systemic blood-derived transcriptomic profiles (PBMCs / whole blood) of young adults aged 18–40 contain distinct, statistically significant gene expression alterations that reliably distinguish young-onset Type 2 Diabetes Mellitus (T2D) patients from normoglycemic healthy controls
* **Plain English** *: Circulating white blood cells act like network monitoring daemons—as they pass through stressed organs, their active gene instructions change, allowing us to detect early diabetes in blood before blood sugar tests spike.
###  Biological Pathomechanism Sub-Hypothesis (H2):
> Differentially expressed genes in young-onset T2D blood samples are significantly enriched in biological pathway networks governing chronic innate immune activation (e.g., TNF-$\alpha$ and NF-$\kappa$B cytokine signaling) and disrupted cellular energy regulation (e.g., the PI3K/Akt insulin signaling cascade).
* **Plain English** *: The specific genes turned ON or OFF in young diabetes patients will cluster around two main cellular problems: constant low-level inflammation and broken insulin communication pipelines.
### Predictive Model Hypothesis
> A feature-selected panel of high-confidence candidate transcripts extracted via dimensional reduction (e.g., LASSO or Recursive Feature Elimination) and trained on a supervised classifier (e.g., Random Forest or Support Vector Machine) will yield strong diagnostic performance (AUC $\ge 0.85$, Sensitivity $\ge 80\%$) in identifying young-onset T2D.
* **Plain English** *: By selecting the most informative genes and training a machine learning model, we can accurately predict which young adults have early-onset T2D.
---

## Repository Structure

```text
young-t2d-bioinformatics/
├── blueprint.md               # Complete project blueprint & checkpoint roadmap
├── AGENT_CONTEXT.md           # Contextual AI agent rules & project guardrails
├── README.md                  # Main repository documentation
├── requirements.txt           # Python virtual environment dependencies
├── notebooks/                 # Jupyter notebooks for development and analysis
│   ├── 00_project_setup.ipynb
│   ├── 01_data_inventory.ipynb
│   ├── 02_qc_preprocessing.ipynb
│   ├── 03_differential_expression.ipynb
│   ├── 04_pathway_enrichment.ipynb
│   ├── 05_biomarker_modeling.ipynb
│   └── README.md
│
├── data/
│   ├── raw/                   # Downloaded GEO SOFT / Matrix files
│   └── processed/             # Normalized, filtered gene expression matrices
│
└── results/
    ├── figures/               # Output boxplots, PCA plots, volcano plots (.png)
    └── tables/                # Differentially expressed gene tables (.csv)
```

---

## Quickstart & Installation

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed on your system.

### 2. Virtual Environment Setup
Clone this repository and set up a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (PowerShell on Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Workflow Execution Guide

Develop the project in the notebook sequence below:

```bash
python -m jupyter lab
```

Notebook order:

1. `notebooks/00_project_setup.ipynb`
2. `notebooks/01_data_inventory.ipynb`
3. `notebooks/02_qc_preprocessing.ipynb`
4. `notebooks/03_differential_expression.ipynb`
5. `notebooks/04_pathway_enrichment.ipynb`
6. `notebooks/05_biomarker_modeling.ipynb`

Each notebook includes starter cells for reproducibility, shared paths, and placeholder analysis steps that write outputs to `data/processed/`, `results/figures/`, and `results/tables/`.

---

## Data Governance & Standards

All datasets utilized in this study are derived from peer-reviewed, open-access public repositories (NCBI GEO / EMBL-EBI ArrayExpress) adhering strictly to MIAME (Minimum Information About a Microarray Experiment) standards.

* **Primary Discovery Accession:** NCBI GEO GSE26168
* **Independent Validation Accession:** NCBI GEO GSE15932
* **Tissue Scope:** Whole Blood / Peripheral Blood Mononuclear Cells (PBMCs)

---

## Team & Acknowledgments

* **Class Group:** E9
* **Track:** Track 6: Applied / Clinical Bioinformatics
* **Institution:** Course Project Submissions
