# Workspace Tracking Audit

Generated: `2026-05-11`

Scope: DOS preview release workspace after the preview-source cleanup through
commit `b7259df`.

## Current Source State

- The working tree was clean before this audit/archive update.
- The repository had `797` tracked files before this audit/archive update.
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
- Verified preview zip hash after the tracked-input guard:
  `54bba902a2cfea0616fc1b07687b41d56a781f2c792b2fbb92762362ab89c2cd`.
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
