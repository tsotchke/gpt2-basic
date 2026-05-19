# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-19T05:54:13Z`
Refreshed: `2026-05-19T06:45:40Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target branch: `main`
Target commit: `273f5e133f2fe5069cc8796931525839cce2e60f`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,841,142 B | `sha256:d130834456e819b47098f7f3a4a62fc89c36d9457f3c87df74a9dc096c9e20f9` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:96b6e5a6030673de8941d573db956545f96da5c9604b42b9f41234417b268f6d` |
| `gpt2-basic-hardware-transfer.zip` | 13,166,526 B | `sha256:626e8bb4a5eb6ccda9bf116fec889dd50bd40a7eaba93f72496137f8588017b3` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:35e20c6be7fe7eb19ab82b3a253f7d9b725677aa05f857a720d05eaab2f54d2b` |
| `gpt2-basic-launch-kit.zip` | 43,451,801 B | `sha256:6d5a2466c9764f3d7e255d9ae203520e7b8ca8ab6916ef49b6ba6bd1b2cff023` |
| `gpt2-basic-launch-kit.zip.sha256` | 92 B | `sha256:f5d6db5aaa02518a593b8c274419b899d450a3002dfb0e339a35339fb07c6b48` |
| `preview_release_manifest.md` | 7,862 B | `sha256:72b3f39cf996f806520145bb886477ef31bb730b08b57c877f2f03ef87e385e4` |
| `gpt2_basic_real_dos_session_1080p.mp4` | 1,429,036 B | `sha256:6c353001fcf09ffed47a47f49cd8ae47597e6b1bfcf9942d2b1e0fd983465c6b` |
| `gpt2_basic_real_dos_session_vertical.mp4` | 506,314 B | `sha256:b602ed12ee41dc17d7d3c683448612ddbba6b568787522b7d2a6a528cb3440da` |
| `gpt2_basic_terminal_demo_1080p.mp4` | 304,149 B | `sha256:f657d37d4865e7887aabbcdc7c22cbfa93f656e06b513ca2d4fa1c3495d7ae20` |
| `gpt2_basic_terminal_demo_vertical.mp4` | 283,619 B | `sha256:76d115057c6e717e88adafe95c6f5c8808ba638ffb048eb6ede5c57e269a6079` |
| `gpt2_basic_launch_teaser_1080p.mp4` | 406,196 B | `sha256:4cf1f45c2e67fcd1af9971ff01d7ced390ca48d294e115b248d70469c378b480` |
| `gpt2_basic_launch_short_vertical.mp4` | 386,724 B | `sha256:06ea06966d09f31ea78ac25f2a533c5fbf9578b54d6fb3e5668c53a3a0a0827f` |
| `thumbnail_gpt_in_dos.png` | 45,242 B | `sha256:5b3b7170e40d23b176796e759caeac4ef44452f5e2f02a91a87a6cb9256ebf27` |

## Verification Before Publish

```sh
python3 -m unittest discover tests
python3 scripts/build_preview_release.py --self-test
python3 scripts/build_hardware_transfer.py --self-test
python3 scripts/build_launch_kit.py --self-test
python3 scripts/build_preview_release.py --force
python3 scripts/build_hardware_transfer.py --force
python3 scripts/build_launch_kit.py --force
python3 scripts/verify_preview_artifacts.py
git diff --check
```

## Remote Verification After Publish

```sh
gh release view v0.1.0-preview -R tsotchke/gpt2-basic --json tagName,isPrerelease,url,targetCommitish,name,assets,publishedAt,createdAt
gh run view 26080991744 -R tsotchke/gpt2-basic --json status,conclusion,headSha
git ls-remote https://github.com/tsotchke/gpt2-basic.git refs/heads/main refs/tags/v0.1.0-preview
```

Remote `refs/heads/main` and `refs/tags/v0.1.0-preview` both pointed to
`273f5e133f2fe5069cc8796931525839cce2e60f` after the final refresh. The public
`Preview Release` workflow completed successfully for that commit.
