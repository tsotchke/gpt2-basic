# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-19T05:54:13Z`
Refreshed: `2026-05-19T09:08:00Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target branch: `main`
Target commit: verify live `refs/heads/main` and `refs/tags/v0.1.0-preview`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,845,361 B | `sha256:c0f5f269d007e23ab004f3097d8b26d0fbd731931d7c6bf958c5a4697539e4a0` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:c6079da42584ea9a0147871ee8570b8e7f8d6ac2728d97339cca5704b973dda6` |
| `gpt2-basic-hardware-transfer.zip` | 13,166,526 B | `sha256:626e8bb4a5eb6ccda9bf116fec889dd50bd40a7eaba93f72496137f8588017b3` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:35e20c6be7fe7eb19ab82b3a253f7d9b725677aa05f857a720d05eaab2f54d2b` |
| `gpt2-basic-launch-kit.zip` | 43,466,239 B | `sha256:50789f184b0df4c56c095f30e4d9c53f93fbfd5ffce955f3cd7a16aa2afa8762` |
| `gpt2-basic-launch-kit.zip.sha256` | 92 B | `sha256:72e103379d5c7bd1ce256329f7251454a359537e431426218790ff3007f4b613` |
| `preview_release_manifest.md` | 7,862 B | `sha256:095733a79b77b10a9b19f7e25cf73df5fa5dc6fc0f228d96af3a94d3a6cd0425` |
| `gpt2_basic_real_dos_session_1080p.mp4` | 1,429,036 B | `sha256:6c353001fcf09ffed47a47f49cd8ae47597e6b1bfcf9942d2b1e0fd983465c6b` |
| `gpt2_basic_real_dos_session_vertical.mp4` | 506,314 B | `sha256:b602ed12ee41dc17d7d3c683448612ddbba6b568787522b7d2a6a528cb3440da` |
| `gpt2_basic_terminal_demo_1080p.mp4` | 304,149 B | `sha256:f657d37d4865e7887aabbcdc7c22cbfa93f656e06b513ca2d4fa1c3495d7ae20` |
| `gpt2_basic_terminal_demo_vertical.mp4` | 283,619 B | `sha256:76d115057c6e717e88adafe95c6f5c8808ba638ffb048eb6ede5c57e269a6079` |
| `gpt2_basic_launch_teaser_1080p.mp4` | 410,222 B | `sha256:fa7f9119b0b3a511244d4b2c552172e2bfc015a3fc5a7990a5310779c030099e` |
| `gpt2_basic_launch_short_vertical.mp4` | 391,193 B | `sha256:cb069aa9418c39fabcd2df908800c954d3a1247d2d7890292a483a3300cf618b` |
| `thumbnail_gpt_in_dos.png` | 45,242 B | `sha256:5b3b7170e40d23b176796e759caeac4ef44452f5e2f02a91a87a6cb9256ebf27` |

## Verification Before Publish

```sh
python3 -m unittest discover tests
python3 scripts/build_preview_release.py --self-test
python3 scripts/build_hardware_transfer.py --self-test
python3 scripts/build_launch_kit.py --self-test
python3 scripts/build_promo_materials.py --self-test
python3 scripts/build_preview_release.py --force
python3 scripts/build_hardware_transfer.py --force
python3 scripts/build_promo_materials.py --force
python3 scripts/build_launch_kit.py --force
python3 scripts/verify_preview_artifacts.py
git diff --check
```

## Remote Verification After Publish

```sh
gh release view v0.1.0-preview -R tsotchke/gpt2-basic --json tagName,isPrerelease,url,targetCommitish,name,assets,publishedAt,createdAt
gh run list -R tsotchke/gpt2-basic --workflow "Preview Release" --limit 1
git ls-remote https://github.com/tsotchke/gpt2-basic.git refs/heads/main refs/tags/v0.1.0-preview
```

Remote `refs/heads/main` and `refs/tags/v0.1.0-preview` must match after the
final refresh. The public `Preview Release` workflow must complete successfully
for that target.
