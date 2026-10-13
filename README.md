# Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus

## Project Overview

This repository contains the complete computational workflow, notebook-based analysis environment, statistical models, and documentation for Group E9's research project: **"Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus"**.

Standard clinical screening tools (HbA1c, fasting blood glucose) flag diabetes only after substantial, irreversible pancreatic beta-cell exhaustion has taken root. This study leverages public blood-derived transcriptomic profiles (Microarray / RNA-seq) from NCBI GEO to identify early molecular markers of systemic cellular stress in young adults (ages 18–40).

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

Each notebook includes starter cells for reproducibility, shared paths, and placeholder analysis steps that write outputs to `dataset/processed`, `results/figures/`, and `results/tables/`.

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
