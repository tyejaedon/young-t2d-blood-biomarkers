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
* **Directory Hygiene:** All scripts must write working outputs to `../dataset/processed`, figures to `results/figures/`, and tabular outputs to `results/tables/`.
* **Logging & Transparency:** Every Python script must print clear execution logs (matrix shapes, missing value counts, variance ratios).
* **Reproducibility:** Fix random seeds explicitly across stochastic operations (`np.random.seed(42)`, `random_state=42` in scikit-learn).

---

## 4. Common Failure Modes to Prevent

* **Failure Mode 1: Pure Machine Learning Focus.** Avoid presenting ML accuracy scores or code without explaining what the candidate genes mean biologically.
* **Failure Mode 2: Unnormalized Data Entry.** Always verify that input matrices are log2-transformed and quantile-normalized before running linear regression or feature selection.
* **Failure Mode 3: Overfitting on Single Dataset.** Always validate prioritized biomarkers on an independent secondary dataset (GSE15932).

---

## 5. Token Economy & User Visibility Rules

1. **Minimize token burn before answering.** Prefer the smallest number of file reads, searches, and edits needed to complete the task correctly.
2. **Use bounded outputs.** Summaries should be concise, grounded in repository evidence, and avoid repeating large code/doc blocks unless explicitly requested.
3. **Expose token cost to the user.** At the start of any non-trivial response, include a short token-burn forecast using one of:
   * `Token Burn: Low` — one file / one direct answer.
   * `Token Burn: Medium` — a few files, light editing, or one validation pass.
   * `Token Burn: High` — broad codebase research, multi-file edits, or repeated validation.
4. **Explain why the burn is necessary.** Add a one-line reason such as `Reason: requires checking preprocessing and downstream notebook dependencies.`
5. **Avoid speculative exploration.** Do not inspect unrelated files “just in case.” Follow dependency chains only when required.
6. **Do not restate context already present in the thread** unless it changes the decision or protects correctness.

---

## 6. Response Contract for Efficient Agent Behavior

For complex work, structure responses in this order:

1. **Acknowledgement:** one short sentence stating the next action.
2. **Checklist:** 3–6 concrete steps only.
3. **Token Burn:** `Low`, `Medium`, or `High` with a reason.
4. **Execution:** perform the work without waiting for confirmation when the task is safe and clear.
5. **Result Summary:** state what changed, where, and any follow-up commands.

Preferred compact status format:

```text
Working on: <task>
Checklist:
- <step 1>
- <step 2>
- <step 3>
Token Burn: <Low|Medium|High>
Reason: <why>
```

---

## 7. Approved Agent Skills

Agents operating in this repository should prioritize these skills in order:

1. **Targeted Search:** find the exact file/symbol first; stop searching once the answer location is known.
2. **Dependency Tracing:** read only the files required to understand inputs, outputs, and side effects.
3. **Minimal Editing:** prefer small patches over full rewrites; preserve current APIs and notebook flow.
4. **Validation:** check edited files for errors and confirm the change did not break expected usage.
5. **Compression:** summarize findings in decision-ready bullets instead of long narrative explanations.
6. **Biological Grounding:** tie computational outputs back to the project’s T2D biomarker question.

---

## 8. Negative Prompting Guardrails (Anti-Drift)

Attach the following constraints to planning or execution prompts whenever the task is non-trivial:

* Do **not** browse broadly once the relevant file or function is identified.
* Do **not** rewrite entire files when a localized patch is sufficient.
* Do **not** provide generic textbook explanations unless the user asks for them.
* Do **not** invent dataset properties, normalization state, or evaluation results without repository evidence.
* Do **not** switch scope from blood-derived young-onset T2D analysis to unrelated disease, tissue, or age cohorts.
* Do **not** spend tokens narrating internal reasoning; spend them on verified actions and concise conclusions.
* Do **not** keep exploring after the acceptance criteria are satisfied and validation passes.

---

## 9. Prompt Snippet for Consistent Agent Execution

Use this compact instruction block when spawning or steering an agent:

```text
Stay within the requested scope. Use the fewest repository reads needed for a correct answer.
Before acting, estimate Token Burn as Low, Medium, or High and state the reason in one line.
Prefer targeted edits, concise summaries, and evidence-backed claims.
Do not drift into unrelated files, speculative refactors, or long explanations.
Stop searching once the relevant implementation path is confirmed.
```

---

## 10. Autonomous Execution & Tool Approval Policy (Full-Allow Mandate)

> **Policy Directive: AUTO_FULL_ALLOW = TRUE**
>
> In accordance with the user's explicit authorization for autonomous execution in `C:/Bioinformatics Project`:

1. **Unconditional Auto-Approval for In-Workspace Tools:**
   * All file operations (`client_edit_file`, `client_create_file`, `client_view_file`) located within `C:/Bioinformatics Project` are pre-authorized and must proceed without prompt-gating.
   * All local python executions and test runners (`run_command` via `.venv/Scripts/python.exe`) within `C:/Bioinformatics Project` are designated as safe and pre-authorized.
2. **No Interactive Prompt Halting:**
   * Agents MUST NOT pause or block execution waiting for redundant user confirmations when running automated data transformations, tests, or validation checks.
   * Agents must proceed directly to execution and report output metrics in the final summary.
3. **Safety Guardrails Under Auto-Allow:**
   * Auto-allow applies strictly within the `C:/Bioinformatics Project` workspace boundary.
   * Operations outside the project root or destructive directory deletions (`rm -rf` / `Remove-Item` on raw source data) remain forbidden.
   * Token consumption remains governed by `TokenGuardMiddleware` thresholds (turn limit: 6, max quota: 120,000 tokens).

---
