# Contributing

GPT2-BASIC is a small, evidence-driven DOS preview. Contributions should keep
the release reproducible and should not blur emulator evidence, generated media,
and physical-machine evidence.

## Before Opening a Pull Request

Run the focused gates that match your change:

```sh
python3 -m unittest discover tests
python3 scripts/verify_workspace_tracking.py
python3 scripts/build_preview_release.py --self-test
python3 scripts/build_hardware_transfer.py --self-test
python3 scripts/build_launch_kit.py --self-test
python3 scripts/verify_preview_artifacts.py
git diff --check
```

Use the project environment that has Torch installed for model-quality and
release-package tests. Use an environment with Pillow plus `ffmpeg` for
`scripts/build_promo_materials.py`; install the Python dependency with:

```sh
python3 -m pip install -r requirements-promo.txt
```

## Evidence Rules

- Do not commit generated release zips, MP4 renders, VM images, emulator
  runtime state, fetched corpus caches, or local operator notes.
- If a change affects the preview package, rebuild it and refresh
  `qemu/evidence/preview_release_manifest.md`.
- If a change affects release media, rebuild the launch kit before updating a
  release.
- Keep local filesystem paths, machine-specific names, and private staging
  repository links out of public docs and evidence.
- Physical-machine claims need returned DOS logs staged through
  `docs/hardware-validation.md`, `scripts/verify_hardware_capture.py`, and
  `scripts/stage_hardware_capture_evidence.py`.

## Scope

Good contributions are small and verifiable: DOS runtime fixes, release gate
improvements, documentation corrections, model-quality evidence, and real
hardware validation. Larger model or runtime changes should include the command
transcript and evidence files needed to reproduce the result.
