# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-19T05:54:13Z`
Refreshed: `2026-05-19T08:55:00Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target branch: `main`
Target commit: verify live `refs/heads/main` and `refs/tags/v0.1.0-preview`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,841,567 B | `sha256:1bc658d57741f897401193ba5ef7963657f524e3a8aa9e9e2b484731c349ef9c` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:55d8b688dde4a54d44b4b0245469f8c4519e5c54892071d1152f4278cbb9bb55` |
| `gpt2-basic-hardware-transfer.zip` | 13,166,526 B | `sha256:626e8bb4a5eb6ccda9bf116fec889dd50bd40a7eaba93f72496137f8588017b3` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:35e20c6be7fe7eb19ab82b3a253f7d9b725677aa05f857a720d05eaab2f54d2b` |
| `gpt2-basic-launch-kit.zip` | 43,460,739 B | `sha256:25e8d8e95cbd8abbc2d8e9c85c63cf368df8066cd4843ceac2d772153372d5c3` |
| `gpt2-basic-launch-kit.zip.sha256` | 92 B | `sha256:5ba7b310b7e84cbfb93961d8dc7c25bd92390a8436bdd3af5fdb785d950d7883` |
| `preview_release_manifest.md` | 7,862 B | `sha256:9fa9ca9a5d2a8adc3a74ea1c626495eb3895273dc6d82cdedec3b1bda52fc2ea` |
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
