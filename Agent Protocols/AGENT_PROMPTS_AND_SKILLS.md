# AGENT_PROMPTS_AND_SKILLS.md — Efficient Agent Rules, Prompts, Skills, and Instructions

This document provides reusable prompt assets for agents working in this repository. The goal is to maximize correctness per token, reduce unnecessary exploration, and keep the user informed about likely token burn before deeper work begins.

---

## 1. Core Operating Rules

### 1.1 Efficiency Rules
- Start narrow, not broad.
- Read the minimum number of files needed to answer correctly.
- Prefer direct evidence from the workspace over general knowledge.
- Use short progress updates instead of repeated explanations.
- Stop once the requested outcome is implemented and validated.

### 1.2 Editing Rules
- Patch the smallest possible region.
- Preserve public interfaces unless the task requires a breaking change.
- Avoid opportunistic refactors during focused tasks.
- Add only the imports, tests, and docs necessary to support the change.

### 1.3 Communication Rules
- Begin with a short acknowledgement and a checklist.
- Always include a **Token Burn** estimate for non-trivial work.
- Report decisions in bullets, not long prose.
- Distinguish verified facts from assumptions.

---

## 2. Token Burn Policy

### 2.1 Required User-Facing Label
Use exactly one of these labels near the top of the response:

- `Token Burn: Low` — simple answer, one file, or one localized change.
- `Token Burn: Medium` — multiple related files, moderate analysis, or one validation cycle.
- `Token Burn: High` — broad search, multi-file implementation, debugging loops, or repeated validation.

### 2.2 Required Reason Line
Immediately follow the label with:

- `Reason: <single sentence explaining the expected work>`

### 2.3 Escalation Rule
If scope expands during execution, explicitly update the user:

```text
Token Burn update: Medium → High
Reason: the change affects downstream notebooks and shared preprocessing logic.
```

### 2.4 Token Savings Tactics
- Avoid quoting full files.
- Avoid duplicate summaries.
- Avoid searching the same concept twice.
- Avoid reading generated data unless needed for validation.
- Prefer one strong summary over multiple incremental restatements.

---

## 3. Approved Skills Catalog

### Skill 1 — Targeted Search
**Use when:** the relevant file is unknown.

**Instruction:**
> Find the narrowest set of files that can answer the request. Stop searching as soon as the ownership of the behavior is clear.

### Skill 2 — Dependency Tracing
**Use when:** a change may impact upstream/downstream logic.

**Instruction:**
> Trace imports, function calls, and output paths only far enough to verify side effects and compatibility.

### Skill 3 — Minimal Patch Editing
**Use when:** the file is already understood.

**Instruction:**
> Make the smallest safe edit that satisfies the request. Do not rewrite surrounding code unless required for correctness.

### Skill 4 — Validation-First Completion
**Use when:** code or config changes were made.

**Instruction:**
> Validate edited files immediately after the change. Fix only relevant breakages and stop when the target behavior is confirmed.

### Skill 5 — Compressed Reporting
**Use when:** returning results to the user.

**Instruction:**
> Summarize what changed, why it changed, and any next step in 3–6 bullets.

### Skill 6 — Scope Guarding
**Use when:** the request could expand into broader redesign.

**Instruction:**
> Keep the implementation bounded to the requested outcome. Defer optional enhancements unless they are necessary for correctness.

---

## 4. Prompt Templates

### 4.1 Master Task Prompt
```text
You are working in a constrained repository. Complete only the requested task.
Use the fewest file reads and edits needed for a correct result.
Start with a short acknowledgement, checklist, and Token Burn label with reason.
Prefer evidence from the workspace over assumptions.
Validate any code or config you change.
Return a concise summary of files changed and remaining risks, if any.
```

### 4.2 Search Prompt
```text
Locate the exact file(s), symbol(s), or document section responsible for the requested behavior.
Search narrowly. Stop once the answer location is confirmed.
Do not summarize unrelated findings.
Return file paths with one-line relevance notes.
```

### 4.3 Editing Prompt
```text
Implement the requested change with the smallest safe patch.
Preserve existing style, APIs, and directory conventions.
Do not perform unrelated cleanup or refactoring.
After editing, validate the modified file(s) and report only meaningful results.
```

### 4.4 Debugging Prompt
```text
Identify the root cause using existing logs, errors, and the smallest necessary code path.
Do not speculate if the workspace can verify the answer.
Fix the actual cause, not nearby symptoms.
Retest after the change and summarize the verified outcome.
```

### 4.5 Documentation Prompt
```text
Write concise documentation that improves execution quality without duplicating repository content.
Prefer checklists, tables, and templates over long narrative blocks.
Include user-visible Token Burn guidance when the document describes agent workflow.
```

---

## 5. Negative Prompting Blocks (Anti-Drift)

Use one or more of these blocks when an agent may wander off task.

### 5.1 General Anti-Drift Block
```text
Negative constraints:
- Do not inspect unrelated files once the target file is identified.
- Do not explain background concepts unless the user asked.
- Do not propose broad refactors.
- Do not repeat file contents in full.
- Do not continue searching after the required evidence is found.
```

### 5.2 Coding Anti-Bloat Block
```text
Negative constraints:
- Do not rewrite entire modules for a local change.
- Do not add abstractions without a demonstrated need.
- Do not introduce new dependencies unless necessary.
- Do not change formatting outside the edited region.
- Do not leave validation unperformed.
```

### 5.3 Research Anti-Speculation Block
```text
Negative constraints:
- Do not infer biological conclusions that are not supported by repository data.
- Do not assume normalization, cohort filtering, or validation status without checking.
- Do not generalize from one dataset to all datasets.
- Do not substitute external intuition for project-specific evidence.
```

### 5.4 Conversation Anti-Token-Waste Block
```text
Negative constraints:
- Do not narrate internal reasoning.
- Do not restate the user's request multiple times.
- Do not produce long prose when bullets will do.
- Do not include optional extras unless they materially improve correctness.
```

---

## 6. Ready-to-Use Prompt Combinations

### 6.1 Fast Code Task
```text
Complete the requested code change with minimal repository reads.
Start with a checklist and Token Burn label.
Use targeted search only if the file is unknown.
Apply the smallest safe patch and validate edited files.
Negative constraints:
- Do not refactor unrelated code.
- Do not browse unrelated folders.
- Do not provide a long explanation.
```

### 6.2 Focused Documentation Task
```text
Create or update documentation needed for the requested workflow.
Use concise headings, checklists, and templates.
Include Token Burn guidance if agents are involved.
Negative constraints:
- Do not duplicate existing README content.
- Do not add speculative policy.
- Do not write long narrative sections when a table or bullets is clearer.
```

### 6.3 Data/Analysis Task
```text
Work strictly within the young-onset human blood transcriptomics scope.
Verify dataset identity, preprocessing status, and output paths before analysis.
Summarize findings with biological interpretation only when supported by evidence.
Negative constraints:
- Do not use non-blood tissue as the primary dataset.
- Do not skip validation on GSE15932 when reporting prioritized biomarkers.
- Do not discuss model performance without the underlying evaluation evidence.
```

---

## 7. Recommended Response Skeleton

```text
Working on: <task>
Checklist:
- <step 1>
- <step 2>
- <step 3>
Token Burn: <Low|Medium|High>
Reason: <why>

Progress/Result:
- <verified finding or completed action>
- <verified finding or completed action>

Changed Files:
- <path> — <what changed>

Validation:
- <check performed>

Next:
- <optional next action or “None”>
```

---

## 8. Example “Stay On Track” Meta-Prompt

```text
Complete the task accurately with the least token cost that still preserves correctness.
Stay within the user's requested scope and the repository's biological constraints.
Search narrowly, edit minimally, validate immediately, and report concisely.
If the task starts expanding, announce the Token Burn increase before continuing.
Do not drift into unrelated files, speculative improvements, or repeated explanations.
```

