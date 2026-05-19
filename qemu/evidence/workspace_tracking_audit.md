# Workspace Tracking Audit

Generated: `2026-05-11`

Scope: DOS preview release workspace after the preview-source cleanup,
workspace-audit archive, hardware-transfer tracked-input guard, and preview
host-path portability guard, plus the workspace tracking verifier.

## Current Source State

- The working tree was clean after commit `d409869` before the hardware-transfer
  tracking guard was added.
- The repository had `798` tracked files after the audit archive update.
- The repository has `801` tracked files after adding the workspace tracking
  verifier and probe evidence.
- The repository has `804` tracked files after adding the real interactive
  QEMU assistant demo launcher, DOS entry batch, and regression tests.
- The repository has `813` tracked files after adding the CHAT conversation
  pack and per-pack usage instructions for CHAT, DOSHELP, and OFFICE.
- The stale pre-cleanup ICC report was archived at
  `qemu/evidence/archive/release_diff_audit_pre_cleanup.md`.
- The current top-level release surface is the DOS preview and hardware-transfer
  bundle; OS/2/Warp remains deferred.

## Tracked Release Inputs

- `qemu/evidence/GPT2.EXE` is tracked because both the preview package and the
  hardware-transfer bundle ship that DOS executable.
- `data/domain_curriculum/domain_curriculum.txt` and
  `data/domain_curriculum/SOURCE_MANIFEST.json` are tracked because the preview
  package ships `data/domain_curriculum/` as rebuild and repair input.
- `scripts/build_preview_release.py` now refuses untracked files in copied
  release-input roots when run from a Git checkout.
- `scripts/build_hardware_transfer.py` now applies the same tracked-input rule
  to model, pack, executable, hardware, and staged-source inputs.
- `scripts/verify_preview_artifacts.py` rejects host absolute path leaks inside
  the preview tree, including POSIX home-directory and Windows user-profile
  fragments.
- `scripts/verify_workspace_tracking.py` rejects untracked files and ignored
  files outside the documented local-runtime/cache buckets.
- `qemu/run_assistant_interactive_486.sh` and
  `qemu/fdauto_assist_interactive.bat` are the product demo path: they launch
  `ASSIST.EXE` without `--scripted`, without redirecting to `ASSIST.LOG`, and
  without powering off QEMU after startup.
- `assets/gpt2_basic/PACKS/CHAT` is the default conversation pack, and every
  listed assistant pack now has a tracked `USAGE.TXT` surfaced through
  `ASSIST.EXE` with `/about`.
- `qemu/make_dos_staging.py` now recreates `qemu/staging/GPT2SRC` from scratch
  before writing staged DOS source files, so stale ignored staging files cannot
  enter the transfer bundle.
- Verified preview zip hash after the tracked-input and host-path guards:
  `29434ce9b0410cdd7eff8ef9caed3004487eed2d14aaa73f744315a13c2746d2`.
- Verified hardware-transfer zip hash:
  `96291d959e33250fd3ac82150ccf2505ab939ce8e41d8b2cc23c4d6bd34c3c72`.

## Intentionally Local

The remaining ignored paths are local runtime, cache, bootstrap, or operator
notes:

- `.DS_Store`, `qemu/.DS_Store`
- `.venv-torch/`, `.venv-torch311/`
- `data/online_corpus/`
- `fbc_build/`, `fbc_win_binary/`
- `memory-bank/`
- `qemu/__pycache__/`, `scripts/__pycache__/`, `tests/__pycache__/`
- `qemu/boot-test.img`, `qemu/gpt2fat.img`, `qemu/gpt2hdd.img`
- `qemu/staging/`
- `third_party/`
- `to_latex.sh`

`data/online_corpus/` stays ignored because it is a fetched network corpus
cache. Its provenance and fetch policy are tracked through
`scripts/fetch_online_training_corpus.py` and
`qemu/evidence/online_training_data_audit.md`.

The current guard probe is stored at
`qemu/evidence/workspace_tracking_probe.log`.
