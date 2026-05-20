# Assistant Generalist Repair Evidence

Date: `2026-05-20`

## Issue

The CHAT pack checkpoint was not good enough as a direct conversational model.
The release assistant could avoid many bad answers through golden replies,
retrieval, memory, and fallback guards, but the pack-local model quality report
still showed `NEEDS_TRAINING` before this repair.

## Fix

- Added explicit training coverage for wrapper-style prompts such as
  `please answer this: ...`, `short answer please: ...`,
  `i typed this question: ...`, `dos chat question: ...`, and
  `help me with this: ...`.
- Added missing general CHAT rows for troubleshooting, trust, long prompts,
  memory, release checklists, offline limits, emulator basics, old hardware,
  and small-task planning.
- Added a dedicated generalist prompt gate so these categories cannot regress
  silently.
- Fine-tuned the CHAT pack-local `486dx2-usable` lexicon checkpoint from the
  current model.

## Results

| Gate | Result |
|---|---:|
| CHAT pack quality | `PASS 160/160 avg 0.999` |
| Raw model prompt gate | `PASS 83/83` |
| Generalist prompt gate | `PASS 24/24` |
| Raw model consistency gate | `PASS 498/498 variants, 83/83 groups` |
| Full unit suite | `88 tests OK` |
| Assistant pack verifier | `PASS` |
| QEMU 486 assistant stress | `PASS 40/40 replies` |
| Preview artifact verifier | `PASS` |
| DOSBox zip test | `PASS` |
| Launch-kit zip test | `PASS` |

## Notes

The QEMU stress probe intentionally exercises the shipped assistant shell, so
its answers come from the safest source available: golden rows, retrieval,
memory, or fallback. The raw prompt and consistency gates evaluate the pack
model directly before those runtime repair paths, which is why they are the
primary proof that the CHAT checkpoint itself improved.
