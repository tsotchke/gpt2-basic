# GPT2-BASIC v0.1.0-preview Clean Download Verification

Verified: `2026-05-19T08:55:00Z`

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
| `gpt2-basic-preview.zip` | 27,841,567 B | `1bc658d57741f897401193ba5ef7963657f524e3a8aa9e9e2b484731c349ef9c` |
| `gpt2-basic-hardware-transfer.zip` | 13,166,526 B | `626e8bb4a5eb6ccda9bf116fec889dd50bd40a7eaba93f72496137f8588017b3` |
| `gpt2-basic-launch-kit.zip` | 43,460,739 B | `25e8d8e95cbd8abbc2d8e9c85c63cf368df8066cd4843ceac2d772153372d5c3` |
| `preview_release_manifest.md` | 7,862 B | `9fa9ca9a5d2a8adc3a74ea1c626495eb3895273dc6d82cdedec3b1bda52fc2ea` |

The three zip hashes match the contents of the downloaded `.sha256` sidecars.

## Extraction

The preview, hardware-transfer, and launch-kit zips were extracted into a
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

The launch-kit archive was also tested directly:

```sh
unzip -t /private/tmp/gpt2-basic-public-download/gpt2-basic-launch-kit.zip
```

Result: no compressed-data errors.

This closes the consumer-style download verification gate for
`v0.1.0-preview`.
