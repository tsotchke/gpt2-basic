# GPT2-BASIC Post-Release Tracking Issues

Created: `2026-05-18T04:47:58Z`

Repository: `Tsotchke-Corporation/gpt2-basic`

Release baseline: `v0.1.0-preview`

## Open Follow-Ups

| Issue | Title | Purpose |
|---:|---|---|
| #1 | Capture physical 486 validation logs | Close the solid-release gate with real 486-class `QUAL.LOG`, `PERF.LOG`, assistant logs, and machine notes. |
| #3 | Capture real-hardware performance matrix | Add measured physical-machine performance evidence for the release model set, starting with a 486-class default-model run. |

## Completed Follow-Ups

| Issue | Title | Closed | Result |
|---:|---|---|---|
| #2 | Verify published preview artifacts from a clean download | `2026-05-18T04:59:53Z` | Published preview assets were downloaded and verified from the release page. Evidence: `qemu/evidence/release_download_verify_v0.1.0_preview.md`. |
| #4 | Plan Windows and OS/2 pack-shell parity | `2026-05-18T05:09:13Z` | Shared pack metadata contract and parity plan were documented in `docs/pack-shell-parity.md` and validated by `scripts/assistant_pack_contract.py`. |

## URLs

- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/1`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/2`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/3`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/4`

These issues deliberately keep the preview tag stable. Follow-up work lands on
`master` and can become a later solid-release or preview-respin milestone after
physical hardware evidence exists.
