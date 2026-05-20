# GPT2-BASIC v0.1.0-preview Clean Download Verification

Verified: `2026-05-20T01:35:49Z`

Repository: `tsotchke/gpt2-basic`

Release: `https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview`

## Download Source

The core release assets were downloaded from the public GitHub release into a
fresh temporary directory, not reused from the local release build output:

```text
/private/tmp/gpt2-basic-public-download
```

Downloaded files:

```text
gpt2-basic-preview.zip
gpt2-basic-preview.zip.sha256
gpt2-basic-dosbox.zip
gpt2-basic-dosbox.zip.sha256
gpt2-basic-hardware-transfer.zip
gpt2-basic-hardware-transfer.zip.sha256
gpt2-basic-launch-kit.zip
gpt2-basic-launch-kit.zip.sha256
preview_release_manifest.md
```

## Checksums

The public download was checked with `openssl dgst -sha256`.

| Asset | Size | SHA-256 |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,797,846 B | `5292414039a7d41697c14fbfb7077eccb1faa7f33f79ec780618608512449d8b` |
| `gpt2-basic-dosbox.zip` | 13,260,577 B | `effb5c5149c7bf5443e8358880ca56541b6169321df00c0a86a582f33824bac9` |
| `gpt2-basic-hardware-transfer.zip` | 13,253,902 B | `51fbc3c8602c8b7d04878a9d08747927d79543ef117e1f879fcb95b8c8f25dd1` |
| `gpt2-basic-launch-kit.zip` | 56,728,951 B | `dd1dd0d1e85a9e8183af1bb2d4ae1c5fc154a12a8c6b4e66af4626f2d7eae067` |
| `preview_release_manifest.md` | 8,284 B | `877e49a1d420509ce44836ba00a3a3a9e785517253ef71c27f3d61465c5f9d87` |

The four zip hashes match the contents of the downloaded `.sha256` sidecars.

## Extraction

The preview, DOSBox, hardware-transfer, and launch-kit zips were extracted into a
separate fresh temporary root:

```text
/private/tmp/gpt2-basic-public-extract
```

## Verifier Commands

The release verifier was run against the extracted preview tree with explicit
paths to the downloaded zips, sidecars, extracted hardware bundle, and
downloaded manifest:

```sh
python3 scripts/verify_preview_artifacts.py \
  --preview-dir /private/tmp/gpt2-basic-public-extract/gpt2-basic-preview \
  --preview-zip /private/tmp/gpt2-basic-public-download/gpt2-basic-preview.zip \
  --preview-zip-sha256 /private/tmp/gpt2-basic-public-download/gpt2-basic-preview.zip.sha256 \
  --hardware-dir /private/tmp/gpt2-basic-public-extract/gpt2-basic-hardware-transfer \
  --hardware-zip /private/tmp/gpt2-basic-public-download/gpt2-basic-hardware-transfer.zip \
  --hardware-zip-sha256 /private/tmp/gpt2-basic-public-download/gpt2-basic-hardware-transfer.zip.sha256 \
  --repo-manifest /private/tmp/gpt2-basic-public-download/preview_release_manifest.md
```

Result:

```text
PROBE_OK preview_artifacts_preview_tree=1
PROBE_OK preview_artifacts_preview_zip=1
PROBE_OK preview_artifacts_preview_zip_payload=1
PROBE_OK preview_artifacts_preview_zip_matches_tree=1
PROBE_OK preview_artifacts_hardware_tree=1
PROBE_OK preview_artifacts_hardware_zip=1
PROBE_OK preview_artifacts_hardware_zip_payload=1
PROBE_OK preview_artifacts_hardware_zip_matches_tree=1
```

The DOSBox and launch-kit archives were also tested directly:

```sh
unzip -t /private/tmp/gpt2-basic-public-download/gpt2-basic-dosbox.zip
unzip -t /private/tmp/gpt2-basic-public-download/gpt2-basic-launch-kit.zip
```

Result: no compressed-data errors.

This closes the consumer-style download verification gate for
`v0.1.0-preview`.
