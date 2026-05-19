# Readiness: `gpt2-basic-aspirational-software`

Repo: `gpt2_basic`  Status: `ready`  Score: `100`

Contract gaps: `0`  Liveness actions: `0`  Liveness scanned: `0` (skipped)  Stub/fallback paths: `0`  External backend deps: `0`  Runtime checks: `2`  Oracle: `complete`  Oracle source: `<repo>/.icc/completion-oracles.json`

## Completion Oracle

| Status | Severity | Criterion | Evidence | Action |
|---|---|---|---:|---|
| PASS | high | `completion_condition` | 1 | `keep` |
| PASS | medium | `checkpoint` | 1 | `keep` |
| PASS | medium | `artifact` | 1 | `keep` |

## Runtime Checks

| Status | Check | Evidence | Action |
|---|---|---:|---|
| PASS | `runtime_evidence_present` | 5 | `keep` |
| PASS | `failure_free` | 0 | `keep` |
