# GPT2-BASIC Public Launch Plan

This document is the working plan for turning the current research/development
tree into a public project that can be pushed, demonstrated, and promoted
without overclaiming real-hardware results.

## Current Public Position

GPT2-BASIC is a DOS-era transformer demonstration: host-trained GPT-style
models exported to a FreeBASIC fixed-point runtime, with QEMU evidence and a
hardware-transfer path for physical DOS machines.

The strongest current public claim is:

> GPT2-BASIC runs small transformer language models in DOS using a BASIC
> fixed-point inference runtime, with reproducible QEMU evidence and curated
> preview-release artifacts.

The CHAT assistant pack currently has:

- Model profile: `486dx2-usable`
- Shape: `3L 64D 4H ctx192 hidden256 vocab4096`
- Fixed-point quality gate: `48/48`, average `1.000`
- Two-sentence CHAT outputs in the generated quality report
- Training performed on Apple Metal/MPS, fixed-point evaluation on CPU
- Runtime artifacts validated by `scripts/model_report.py --strict`

Avoid claiming physical 486 performance until we have captured and verified
real-hardware logs. Say "QEMU 486DX2/66 evidence" or "486-era target" unless a
specific physical machine has produced evidence.

## Public Repository Strategy

There are already two configured remotes:

- `origin`: `https://github.com/tsotchke/gpt2-basic.git`
- `company`: `https://github.com/Tsotchke-Corporation/gpt2-basic.git`

Canonical public home: `origin`.

The company repository remains the private staging/review surface. Publish the
public preview from a clean `origin/main` snapshot so old private staging history
does not become part of the public repository. Keep rendered videos, raw
captures, and large editing assets out of git; publish them through GitHub
Releases, YouTube, or a media CDN. Keep release zips and hardware-transfer zips
as GitHub Release assets, not normal source-tree files.

## Publish Gate

Before the first public push or public prerelease, run:

```sh
python3 -m unittest discover tests
python3 scripts/audit_exported_models.py --self-test
python3 scripts/train_assistant_pack_models.py --self-test
python3 scripts/verify_assistant_packs.py
python3 scripts/verify_workspace_tracking.py
python3 scripts/build_preview_release.py --force
python3 scripts/build_hardware_transfer.py --force
python3 scripts/verify_preview_artifacts.py
python3 scripts/build_promo_materials.py --self-test
python3 scripts/build_promo_materials.py --force
python3 scripts/build_launch_kit.py --self-test
python3 scripts/build_launch_kit.py --force
git diff --check
```

For the latest CHAT model specifically, also run:

```sh
python3 scripts/model_report.py --model-dir assets/gpt2_basic/PACKS/CHAT/MODEL --strict
```

If any generated release artifact changes, rebuild the corresponding evidence
and update the release notes before pushing.

The preview-release builder preserves source evidence in the repository, but it
sanitizes local repository paths in selected copied text evidence so published
release zips do not contain machine-specific home-directory paths.

## What Should Ship in Git

Keep these in the public repository:

- FreeBASIC source under `src/`
- Host scripts under `scripts/`
- Tests under `tests/`
- Small documentation and evidence files
- Curated model artifacts that are part of the reproducible preview release
- Assistant pack metadata, help text, icons, sprites, and validated model files
- Release manifests and verification reports

Keep these out of normal git history:

- Raw video capture files
- Rendered promotional videos
- Local VM disks, ISO files, and emulator runtime images
- Temporary screenshots and editor project caches
- Unreviewed online-corpus downloads
- Any credentials, API keys, private notes, or unlicensed third-party media

## Public Readiness Checklist

- [x] Decide canonical remote: `origin`.
- [x] Decide first public branch name: `main`.
- [ ] Confirm README headline and current status match the latest CHAT pack.
- [ ] Confirm LICENSE is acceptable for public release.
- [ ] Run the publish gate above.
- [ ] Build preview and hardware-transfer zips.
- [ ] Verify `qemu/evidence/preview_release_manifest.md`.
- [ ] Capture one clean QEMU demo video.
- [ ] Capture one real-hardware teaser if hardware is available.
- [ ] Build `/private/tmp/gpt2-basic-launch-kit.zip`.
- [ ] Publish public repository.
- [ ] Create a prerelease using `docs/releases/v0.1.0-preview.md`.
- [ ] Attach release zips and checksums.
- [ ] Publish launch video and short clips.
- [ ] Add links from README to the release and video.

## First Public Release Message

Use this as the release thesis:

> GPT2-BASIC is a working DOS/FreeBASIC transformer runtime. It is not a modern
> LLM squeezed into DOS; it is a deliberately small, inspectable GPT-style model
> that proves the core transformer loop can run in a 486-era software stack.

The important distinction is that the project is real inference, but modest
scale. That honesty makes the demo stronger.

## Immediate Next Work

1. Improve real-hardware experience: quiet prefill, compact prompts, optional
   continue mode, and smaller output shortlist sweeps.
2. Refresh README status so CHAT mode is represented as a current assistant
   pack, not only a side experiment.
3. Capture a clean live QEMU/terminal recording to pair with the generated
   ffmpeg starter assets.
4. Build a clean `v0.1.0-preview` prerelease and verify downloaded artifacts.
