# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-19T05:54:13Z`
Refreshed: `2026-05-19T22:23:09Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target branch: `main`
Target commit: verify live `refs/heads/main` and `refs/tags/v0.1.0-preview`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,854,876 B | `sha256:08930e9bbc0b9f0d46ac04059c37c5696b41dcc295e7a06fd9ab75e34d4dbaf3` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:752a50f2676c632c9e1e3690716f8e0f7f1c5b647bb3c21c53fdc47981df4df4` |
| `gpt2-basic-dosbox.zip` | 13,335,578 B | `sha256:4b3ccbbb9c4904d1c02e9a55752d75bdfb5333b1a50d0d5a6b9bfb1b45ddf7b5` |
| `gpt2-basic-dosbox.zip.sha256` | 88 B | `sha256:e7f5b2e1f4ab214c7455dd524879743d4bec747d320a587209d4cbbd6671f4f1` |
| `gpt2-basic-hardware-transfer.zip` | 13,328,823 B | `sha256:822cee2ba80d3643b6b96065303600c77d7774960badcf7ee4ba8541e3a7273a` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:e677bce0277bbc5046828fabac28a35da690bb3d2a1c82f437b3a67b91d005fb` |
| `gpt2-basic-launch-kit.zip` | 56,684,708 B | `sha256:a78eb7b813f422e077c2f86a0f9cb1fb7a0d3f5eb2495d5efabf16b757b1ffdc` |
| `gpt2-basic-launch-kit.zip.sha256` | 92 B | `sha256:9c1d1586faaf4e0d3a313da88b9e19da8a78a2063d1adc9a030c352fb8b76945` |
| `preview_release_manifest.md` | 8,088 B | `sha256:e39a96b72857b148d729b17532cf172a17fe3fa0f2810bc68e5da9435b681506` |
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
