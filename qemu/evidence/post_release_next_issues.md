# GPT2-BASIC Post-Release Tracking Issues

Created: `2026-05-18T04:47:58Z`

Repository: `Tsotchke-Corporation/gpt2-basic`

Release baseline: `v0.1.0-preview`

## Open Follow-Ups

| Issue | Title | Purpose |
|---:|---|---|
| #1 | Capture physical 486 validation logs | Close the solid-release gate with real 486-class `QUAL.LOG`, `PERF.LOG`, assistant logs, and machine notes. |
| #2 | Verify published preview artifacts from a clean download | Prove the GitHub prerelease assets verify from a consumer-style download path, not only local `/private/tmp` build outputs. |
| #3 | Capture real-hardware performance matrix | Add measured physical-machine performance evidence for the release model set, starting with a 486-class default-model run. |
| #4 | Plan Windows and OS/2 pack-shell parity | Keep later Windows and OS/2 shells aligned with the same pack metadata contract used by DOS. |

## URLs

- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/1`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/2`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/3`
- `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/4`

These issues deliberately keep the preview tag stable. Follow-up work lands on
`master` and can become a later solid-release or preview-respin milestone after
physical hardware evidence exists.
