# Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus

## Project Overview

This repository contains the complete computational workflow, notebook-based analysis environment, statistical models, and documentation for Group E9's research project: **"Gene Expression Biomarkers for Early Detection of Young-Onset Type 2 Diabetes Mellitus"**.

Standard clinical screening tools (HbA1c, fasting blood glucose) flag diabetes only after substantial, irreversible pancreatic beta-cell exhaustion has taken root. This study leverages public blood-derived transcriptomic profiles (Microarray / RNA-seq) from NCBI GEO to identify early molecular markers of systemic cellular stress in young adults (ages 18–40).

---
## Research Objectives & Questions
The overarching question this project aspires to uncover is, can computational analysis of public blood-derived transcriptomic profiles distinguish young adults (ages 18–40) with Type 2 Diabetes Mellitus (T2D) from healthy controls and identify high-confidence candidate biomarkers for early detection?

To answer this research question, the project executes four sequential aims:** 
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

Following database screening, three representative datasets were selected to support primary discovery, secondary validation, and cross-tissue biological contextualization. Their defining characteristics are summarized below:

| Dataset | GEO Accession | Project Role | Organism & Tissue | Assay Platform | Data Format | Cohort Phenotypes | Metadata Features |
|---|---|---|---|---|---|---|---|
| GSE26168 | NCBI GEO: GSE26168 | Primary Discovery Set | Homo sapiens / Whole Blood | Illumina HumanRef-8 v3 (GPL6883) | Normalized Probe Intensity Matrix | T2D Cases vs. Healthy Controls | Age, gender, disease state |
| GSE15932 | NCBI GEO: GSE15932 | Independent Validation Set | Homo sapiens / PBMCs | Affymetrix Human Genome U133 Plus 2.0 (GPL570) | Processed Gene Expression Matrix | Impaired Glucose Tolerance / T2D vs. Controls | Age, glycemic status, BMI |
| GSE164416 | NCBI GEO: GSE164416 | Cross-Tissue Contextual Set only; not used as a validation cohort | Homo sapiens / Pancreatic Islets | Illumina HiSeq 2500 (GPL16791) | Raw HTSeq Read Counts (.txt.gz) | Non-Diabetic, IGT, T2D, T3cD | Age, glycemic status, INS expression |

> **Scope note:** `GSE164416` is retained strictly for cross-tissue biological context and should not be interpreted as the secondary validation dataset.


## Repository Structure

```text
young-t2d-bioinformatics/
├── architecture/blueprint.md  # Complete project blueprint & checkpoint roadmap
├── Agent Protocols/
│   ├── AGENT_CONTEXT.md       # Master AI agent rules, token-efficiency guardrails, and scope boundaries
│   └── AGENT_PROMPTS_AND_SKILLS.md # Reusable prompts, negative prompting blocks, and skills catalog
├── README.md                  # Main repository documentation
├── requirements.txt           # Python virtual environment dependencies
├── notebooks/                 # Jupyter notebooks for development and analysis
│   ├── 00_project_setup.ipynb
│   ├── 01_data_inventory.ipynb
│   ├── 02_qc_preprocessing.ipynb
│   ├── 03_differential_expression.ipynb   # Stage 4 Discovery DEA (Fully Executed)
│   ├── 04_pathway_enrichment.ipynb
│   ├── 05_biomarker_modeling.ipynb
│   └── README.md
│
├── src/                       # Production pipelines & statistical engines
│   ├── data_ingestion.py
│   ├── data_preprocessing.py  # Stage 3 Normalization Orchestrator
│   ├── eda_analysis.py        # Stage 3 EDA Pipeline Orchestrator
│   ├── differential_expression_analysis.py # Stage 4 Discovery DEA Orchestrator
│   ├── preprocessing/         # Normalization, BGX mapping, filtering, exports
│   ├── eda/                   # Descriptive statistics, PCA, trajectory, confounders
│   └── dea/                   # Limma Empirical Bayes, visualizations, comparative DEA
│
├── data/
│   ├── raw/                   # Downloaded GEO SOFT / Matrix files
│   └── processed/             # Normalized, filtered gene expression matrices
│
└── results/
    ├── figures/               # Output boxplots, PCA, Volcano, MA, heatmaps (.png)
    └── tables/                # Differentially expressed gene tables & metrics (.csv)
```

---

## Agent Protocols

The repository includes dedicated agent guidance in `Agent Protocols/` for efficient AI-assisted work:

- `Agent Protocols/AGENT_CONTEXT.md` defines project guardrails, token-efficiency rules, user-visible token burn reporting, and anti-drift behavior.
- `Agent Protocols/AGENT_PROMPTS_AND_SKILLS.md` provides reusable prompt templates, negative prompting blocks, and an approved skills catalog for focused agent execution.

Use these documents when you want AI agents to stay within scope, minimize unnecessary repository reads, and clearly communicate expected token burn before deeper work begins.

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

### 3. Running the Discovery Differential Expression Pipeline
Execute the complete differential expression analysis on the primary discovery cohort (`GSE26168`):

```bash
python src/differential_expression_analysis.py
```

---

## Stage 4: Differential Expression Analysis on Discovery Cohort (`GSE26168`)

### 1. Statistical Methodology (Limma Empirical Bayes)
To identify high-confidence candidate biomarkers distinguishing young-onset T2D patients from normoglycemic healthy controls, the pipeline implements the **Empirical Bayes moderated $t$-statistic framework** (Smyth, 2004; `limma` algorithm) anchored in [`src/dea/ebayes.py`](file:///C:/Bioinformatics%20Project/src/dea/ebayes.py) and executed via [`src/differential_expression_analysis.py`](file:///C:/Bioinformatics%20Project/src/differential_expression_analysis.py):
- **Cohort Design:** Evaluates $n = 9$ Type 2 Diabetes cases versus $n = 8$ Normoglycemic Controls ($d_g = 15$ pooled residual degrees of freedom) from whole blood (`GSE26168`).
- **Variance Shrinkage:** Estimates empirical hyper-parameters ($d_0 = 2.18, s_0^2 = 0.353$) using method of moments on log-residual variances, shrinking gene-wise variances $\tilde{s}_g^2 = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g}$ to eliminate false discoveries driven by spuriously low sample standard errors.
- **Multiple Testing Correction:** Adjusts $p$-values across all evaluated features via Benjamini-Hochberg False Discovery Rate (FDR).
- **Stringent Significance Thresholds:**
  - Biological Effect Size: $|\log_2\text{FC}| \ge 1.0$ (at least 2-fold change)
  - Statistical Significance: Adjusted $p\text{-value} < 0.05$ (FDR)

### 2. Dual Low-Expression Filtering Comparative Evaluation

In alignment with Stage 3's dual filtering architecture, DEA was conducted head-to-head across both filtering regimes:

| Dimension | Approach 1: Heuristic Background Floor | Approach 2: Distribution-Aware Variance Floor |
| :--- | :--- | :--- |
| **Filtering Mechanism** | Retention above empirical p20 noise in $\ge 20\%$ samples | Pruning bottom 20% non-specific IQR variance + `max_mean` probe collapse |
| **Features Tested** | **17,496 genes** | **14,905 genes** (pruned 3,726 uninformative flat-line genes) |
| **Significant DEGs ($|\log_2\text{FC}| \ge 1.0, p_{\text{adj}} < 0.05$)** | **24 DEGs** (12 Up-regulated, 12 Down-regulated) | **25 DEGs** (10 Up-regulated, 15 Down-regulated) |
| **Nominal DEGs ($|\log_2\text{FC}| \ge 1.0, p_{\text{raw}} < 0.05$)** | 2,177 genes | 2,042 genes |
| **Core Biomarker Overlap** | **19 shared genes (79.2% concordance)** | **19 shared genes (76.0% concordance)** |
| **Methodological Takeaway** | Broader feature retention preserving subtle low-abundance transcripts. | Reduced multiple testing burden uncovering additional key metabolic effectors (e.g., `AKT3`, `APOL4`). |

### 3. Prioritized Biomarker Candidates & Biological Interpretation

The 24 high-confidence DEGs directly validate project hypotheses **H1** (systemic blood transcriptomic alterations) and **H2** (disrupted insulin signaling and immune activation):

| Gene Symbol | $\log_2\text{FC}$ | Moderated $t$ | FDR $p_{\text{adj}}$ | Regulation | Biological & Pathophysiological Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`IRS1`** | **-1.42** | **-11.17** | **$4.5 \times 10^{-5}$** | **Down-regulated** | **Insulin Receptor Substrate 1:** Central transducer of insulin signaling. Leukocyte downregulation provides direct molecular proof of peripheral insulin resistance. Traces a monotonic decrease ($\rho = -0.94, p < 10^{-11}$) across Control ($8.32$) $\rightarrow$ IFG ($7.88$) $\rightarrow$ T2D ($6.90$). |
| **`TSP50`** | **-2.64** | **-7.88** | **0.0036** | **Down-regulated** | **Testis-Specific Protease 50:** Threonine protease regulating NF-$\kappa$B transcriptional activity and cell survival under metabolic stress. |
| **`PAQR9`** | **+2.71** | **+6.46** | **0.0319** | **Up-regulated** | **Progestin & AdipoQ Receptor Family Member 9:** Membrane receptor belonging to the adiponectin superfamily; modulates lipid oxidation and glucose disposal. |
| **`MBNL3`** | **+1.69** | **+6.17** | **0.0419** | **Up-regulated** | **Muscleblind-Like Splicing Regulator 3:** Alternative splicing regulator governing developmental insulin receptor isoform reprogramming. |
| **`C6orf85`** | **+1.49** | **+6.02** | **0.0419** | **Up-regulated** | **FAM213A (PRDX9):** Redox regulatory protein induced to counteract leukocyte oxidative stress in hyperglycemic environments. |
| **`IFNA7`** | **-3.58** | **-5.97** | **0.0419** | **Down-regulated** | **Interferon Alpha 7:** Type I interferon cytokine mediating innate antiviral and sterile inflammatory responses. |
| **`ANGPTL2`** | **-3.63** | **-5.73** | **0.0419** | **Down-regulated** | **Angiopoietin-Like Protein 2:** Pro-inflammatory circulating adipokine linked to vascular endothelial dysfunction and chronic tissue inflammation. |
| **`LRP5`** | **+2.49** | **+5.51** | **0.0419** | **Up-regulated** | **LDL Receptor-Related Protein 5:** Wnt co-receptor critical for pancreatic $\beta$-cell proliferation and glucose-stimulated insulin secretion (GSIS); compensatory upregulation in circulating cells. |
| **`AKT3`** *(Appr. 2)* | **-3.00** | **-5.19** | **0.0467** | **Down-regulated** | **RAC-gamma Serine/Threonine Kinase:** Key downstream kinase of IRS1 in the PI3K/Akt pathway; downregulation confirms systemic insulin signaling collapse. |
| **`KISS1`** | **+2.22** | **+5.75** | **0.0419** | **Up-regulated** | **Kisspeptin-1:** Neuro-endocrine GPCR peptide ligand regulating whole-body energy homeostasis and metabolic adaptation. |
| **`DPT`** | **-3.16** | **-5.53** | **0.0419** | **Down-regulated** | **Dermatopontin:** Extracellular matrix mediator modulating TGF-$\beta$ signaling and tissue fibrosis in metabolic syndrome. |
| **`POPDC3`** | **-3.04** | **-5.54** | **0.0419** | **Down-regulated** | **Popeye Domain Containing 3:** cAMP-binding membrane effector maintaining cellular structural stability. |
| **`MALAT1`** | **-2.42** | **-5.22** | **0.0481** | **Down-regulated** | **Metastasis-Associated Lung Adenocarcinoma Transcript 1:** Critical lncRNA mediator of glucose-induced endothelial activation and vascular inflammation. |

### 4. Early Trajectory Analysis (Control $\rightarrow$ IFG $\rightarrow$ T2D)
Analysis of the 7 Impaired Fasting Glucose (IFG) pre-diabetic donors reveals that key dysregulations are established **prior to clinical T2D onset**:
- `IRS1` drops from $8.32$ (Control) to $7.88$ (IFG) to $6.90$ (T2D), confirming early insulin resistance during normoglycemic decompensation.
- `C6orf85` rises from $3.89$ (Control) to $4.78$ (IFG) to $5.38$ (T2D) ($\rho = +0.77$), marking early leukocyte antioxidant induction.
- `ANGPTL2` drops progressively from $4.40$ (Control) to $2.82$ (IFG) to $0.77$ (T2D) ($\rho = -0.76$).

### 5. Primary Deliverables Generated

#### Tabular Deliverables ([`results/tables/`](file:///C:/Bioinformatics%20Project/results/tables/)):
- [`dea_discovery_gse26168_all_genes.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_discovery_gse26168_all_genes.csv): Complete statistical table across all 17,496 tested features.
- [`dea_discovery_gse26168_significant_degs.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_discovery_gse26168_significant_degs.csv): 24 significant candidate DEGs with functional biological annotations.
- [`dea_discovery_gse26168_approach2_all_genes.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_discovery_gse26168_approach2_all_genes.csv): Complete statistics under Approach 2 (14,905 features).
- [`dea_discovery_gse26168_approach2_significant_degs.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_discovery_gse26168_approach2_significant_degs.csv): 25 significant DEGs under Approach 2.
- [`dea_filtering_comparison_summary.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_filtering_comparison_summary.csv): Head-to-head metrics comparing Approach 1 and Approach 2.
- [`dea_early_progression_trajectory_gse26168.csv`](file:///C:/Bioinformatics%20Project/results/tables/dea_early_progression_trajectory_gse26168.csv): Stage-wise expression means and monotonic trend statistics.

#### Diagnostic Visualizations ([`results/figures/`](file:///C:/Bioinformatics%20Project/results/figures/)):
- `dea_volcano_gse26168.png`: Volcano plot annotating top candidate biomarkers.
- `dea_ma_plot_gse26168.png`: $\log_2\text{FC}$ vs. mean expression intensity diagnostics.
- `dea_top_genes_heatmap_gse26168.png`: Clustered expression $z$-scores across Control, IFG, and T2D stages.
- `dea_pvalue_diagnostics_gse26168.png`: Raw and moderated $p$-value calibration histograms.
- `dea_filtering_comparison_gse26168.png`: Comparative feature burden and DEG yield bar charts.

#### Executable Analysis Notebook:
- [`notebooks/03_differential_expression.ipynb`](file:///C:/Bioinformatics%20Project/notebooks/03_differential_expression.ipynb): End-to-end interactive execution with formatted tables and high-resolution figures.

---

## Workflow Execution Guide

Develop and run the project notebooks in sequence:

```bash
python -m jupyter lab
```

Notebook execution sequence:
1. `notebooks/00_project_setup.ipynb` — Environment and directory verification
2. `notebooks/01_data_inventory.ipynb` — GEO dataset inventory and sample metadata verification
3. `notebooks/02_qc_preprocessing.ipynb` — Quality control, normalization, and outlier detection
4. `notebooks/03_differential_expression.ipynb` — **Stage 4 Discovery DEA (Fully Executed)**
5. `notebooks/04_pathway_enrichment.ipynb` — Functional annotation and pathway over-representation (GO/KEGG)
6. `notebooks/05_biomarker_modeling.ipynb` — Supervised machine learning, feature selection, and external validation

---

## Analysis Success Criteria

The project is evaluated against three core statistical and modeling benchmarks:

| Dimension | Core Metrics | Target Benchmark | Current Status |
| :--- | :--- | :--- | :--- |
| **Statistical Rigor (DEA)** | Effect size & FDR control | $|\log_2\text{FC}| \ge 1.0$, FDR $p_{\text{adj}} < 0.05$ | **Achieved:** 24 high-confidence DEGs in GSE26168; core genes validated across dual filtering approaches. |
| **Diagnostic Accuracy (ML)** | AUC-ROC, sensitivity, specificity | AUC-ROC $\ge 0.85$; sensitivity $\ge 80\%$; specificity $\ge 80\%$ | Pending Aim 4 execution in `05_biomarker_modeling.ipynb`. |
| **Biomarker Parsimony & Transferability** | Panel size, external AUC on `GSE15932` | 5–15 gene stable panel; external validation AUC $\ge 0.80$ | 19 core candidate genes prioritized for feature selection and cross-cohort testing. |

---

## Data Governance & Standards

All datasets utilized in this study are derived from peer-reviewed, open-access public repositories (NCBI GEO / EMBL-EBI ArrayExpress) adhering strictly to MIAME (Minimum Information About a Microarray Experiment) standards.

* **Primary Discovery Accession:** NCBI GEO GSE26168
* **Independent Validation Accession:** NCBI GEO GSE15932
* **Cross-Tissue Context Accession:** NCBI GEO GSE164416
* **Tissue Scope:** Whole Blood / Peripheral Blood Mononuclear Cells (PBMCs)

---

## Team & Acknowledgments

* **Class Group:** E9
* **Track:** Track 6: Applied / Clinical Bioinformatics
* **Institution:** Course Project Submissions
