# GPT2-BASIC v0.1.0-preview Publish Evidence

Published: `2026-05-18T04:34:36Z`

Repository: `Tsotchke-Corporation/gpt2-basic`

Release: `https://github.com/Tsotchke-Corporation/gpt2-basic/releases/tag/v0.1.0-preview`

Tag: `v0.1.0-preview`

Target commit: `95f6f9f71cc660a851383ec33d4441898d4d4290`

Prerelease: `true`

## Uploaded Assets

| Asset | Size | GitHub digest |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,400,459 B | `sha256:b6ae31b82dcef92cc8be48f4162c9c85c53765b388bd48366cb3e6cb45fabfa1` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `sha256:3a7e82f22c20808381fb10ba9a8c2c1e691ad9e8a85deb9c3b52483ba3224c62` |
| `gpt2-basic-hardware-transfer.zip` | 12,766,204 B | `sha256:59bac4e53222a5513dc907bfa142d9b46a8db209e4d1c7c2ede8dc0309e9e55c` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `sha256:a7487c218d736f8111e09eeab64e754e25370d1398411e40c66a2081c64c2225` |
| `preview_release_manifest.md` | 7,381 B | `sha256:60d513040507f0315ca2f06017dbbdb4ad02a7cc3a0936507079f06259c7e458` |

## Local Verification Before Publish

```sh
python3 scripts/build_preview_release.py --force
python3 scripts/build_hardware_transfer.py --force
python3 scripts/verify_preview_artifacts.py
python3 scripts/verify_assistant_packs.py
```

## Remote Verification After Publish

```sh
gh release view v0.1.0-preview -R Tsotchke-Corporation/gpt2-basic --json tagName,isPrerelease,url,targetCommitish,name,assets,publishedAt,createdAt
git ls-remote https://github.com/Tsotchke-Corporation/gpt2-basic.git HEAD refs/heads/master refs/tags/v0.1.0-preview
```

Remote `HEAD`, `refs/heads/master`, and `refs/tags/v0.1.0-preview` all pointed
to `95f6f9f71cc660a851383ec33d4441898d4d4290` immediately after publish.
