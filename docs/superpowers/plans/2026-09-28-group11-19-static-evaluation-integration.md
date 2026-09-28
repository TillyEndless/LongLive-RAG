# Group11–19 Static Evaluation Integration Implementation Plan

> **For agentic workers:** Execute this plan inline in the isolated worktree only. No inference, GPU work, or active-worktree access.

**Goal:** Add a static Group11–19 experiment registry, artifact adapter, validity/action planner, and coverage CLI to the 5090 unified evaluation pipeline.

**Architecture:** `experiment_registry.py` stores only source/config semantics. `static_audit.py` discovers existing artifacts, classifies provenance and compatibility, normalizes availability, and returns minimum actions without executing experiments. `unified_pipeline.py` exposes the audit command and writes small JSON/CSV/Markdown coverage outputs.

**Tech Stack:** Python standard library, existing unified pipeline/schema, pytest-free static tests compatible with the repository test setup.

---

### Task 1: Registry

**Files:** Create `evaluation/experiment_registry.py`; test `tests/test_evaluation_static.py`.

- Define all 18 experiment IDs with explicit Group11–19 semantics, config/source paths, backend provenance, sparse retained/dropped ratios, storage policy, and `NOT_APPLICABLE`/`NOT_IDENTIFIABLE`/`UNVERIFIED` markers.
- Keep registry free of measured result values.
- Expose `REGISTRY`, `resolve_experiment`, `iter_experiments`, and `registry_to_dict`.

### Task 2: Static artifact adapter and planner

**Files:** Create `evaluation/static_audit.py`; test `tests/test_evaluation_static.py`.

- Discover known artifact roots and classify hardware from paths.
- Return found/path/format/provenance/reusable/reason records for quality, memory, latency, RAG, correctness, and provenance.
- Reject legacy or unversioned latency as reusable when timer boundaries are not explicit.
- Normalize measured fields only when present; never infer missing measurements.
- Return the requested minimum action enums, with Group19 disabled/unresolved mapped to provenance audit.

### Task 3: Coverage CLI and outputs

**Files:** Modify `evaluation/unified_pipeline.py`; create static JSON/CSV/Markdown outputs.

- Add `--audit-groups` and optional hardware filtering without changing normal config execution.
- Generate complete rows, including incomplete/invalid rows.

### Task 4: Tests and documentation

**Files:** Create `tests/test_evaluation_static.py`; modify `docs/unified_evaluation_pipeline.md`.

- Test registry semantics, artifact compatibility, planner actions, and incomplete-row preservation using tiny temporary files only.
- Document static audit behavior and non-execution guarantees.

### Task 5: Verification and publication

- Run `py_compile`, CLI help, static audit CLI, focused tests, `git diff --check`, and repository/process safety checks.
- Inspect the diff and generated output sizes.
- Commit only source/tests/docs/static coverage outputs and push the isolated branch with the configured SSH alias.
