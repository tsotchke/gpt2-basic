# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-19T05:54:13Z`
Refreshed: `2026-05-20T01:35:49Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target branch: `main`
Target commit: verify live `refs/heads/main` and `refs/tags/v0.1.0-preview`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,797,846 B | `sha256:5292414039a7d41697c14fbfb7077eccb1faa7f33f79ec780618608512449d8b` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:7f465097c05cdd9958df13840067c94e9fb87d813fd663f6e5499c7da5dd6493` |
| `gpt2-basic-dosbox.zip` | 13,260,577 B | `sha256:effb5c5149c7bf5443e8358880ca56541b6169321df00c0a86a582f33824bac9` |
| `gpt2-basic-dosbox.zip.sha256` | 88 B | `sha256:b024e66270bb05bd8daa9582367a3a52394da459850f40d839e33c5a8b3534e5` |
| `gpt2-basic-hardware-transfer.zip` | 13,253,902 B | `sha256:51fbc3c8602c8b7d04878a9d08747927d79543ef117e1f879fcb95b8c8f25dd1` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:bc2687df9191e3e6051bdc52fc9f6cec1428b08367e2457aeb4ac67d066f0afb` |
| `gpt2-basic-launch-kit.zip` | 56,728,951 B | `sha256:dd1dd0d1e85a9e8183af1bb2d4ae1c5fc154a12a8c6b4e66af4626f2d7eae067` |
| `gpt2-basic-launch-kit.zip.sha256` | 92 B | `sha256:097fd920e260cde21fa9a5764bddca8eeca160366909188a250b7baff9fcf143` |
| `preview_release_manifest.md` | 8,284 B | `sha256:877e49a1d420509ce44836ba00a3a3a9e785517253ef71c27f3d61465c5f9d87` |
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
python3 scripts/build_dosbox_bundle.py --self-test
python3 scripts/build_hardware_transfer.py --self-test
python3 scripts/build_launch_kit.py --self-test
python3 scripts/build_promo_materials.py --self-test
python3 scripts/build_preview_release.py --force
python3 scripts/build_dosbox_bundle.py --force
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
