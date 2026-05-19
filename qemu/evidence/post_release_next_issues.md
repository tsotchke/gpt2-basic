# GPT2-BASIC Post-Release Tracking Issues

Created: `2026-05-19T06:49:00Z`

Repository: `tsotchke/gpt2-basic`

Release baseline: `v0.1.0-preview`

## Open Follow-Ups

| Issue | Title | Purpose |
|---:|---|---|
| #1 | Capture physical 486 validation logs | Close the solid-release gate with real 486-class `QUAL.LOG`, `PERF.LOG`, assistant logs, and machine notes. |
| #2 | Capture real-hardware performance matrix | Add measured physical-machine performance evidence for the release model set, starting with a 486-class default-model run. |

## Completed Preview-Launch Gates

| Gate | Result |
|---|---|
| Public repository | `https://github.com/tsotchke/gpt2-basic`, default branch `main` |
| Public prerelease | `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview` |
| Public release workflow | Passing: `https://github.com/tsotchke/gpt2-basic/actions/runs/26080991744` |
| Clean download verification | Passed; see `qemu/evidence/release_download_verify_v0.1.0_preview.md` |
| Branch hygiene | Stale public `master` removed; protected `main` requires `Host release gates` |

## URLs

- `https://github.com/tsotchke/gpt2-basic/issues/1`
- `https://github.com/tsotchke/gpt2-basic/issues/2`

These issues deliberately keep the preview tag stable. Follow-up work lands on
`main` and can become a later solid-release or preview-respin milestone after
physical hardware evidence exists.
