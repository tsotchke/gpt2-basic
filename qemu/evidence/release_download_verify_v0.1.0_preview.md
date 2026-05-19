# GPT2-BASIC v0.1.0-preview Clean Download Verification

Verified: `2026-05-19T09:08:00Z`

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
| `gpt2-basic-preview.zip` | 27,845,361 B | `c0f5f269d007e23ab004f3097d8b26d0fbd731931d7c6bf958c5a4697539e4a0` |
| `gpt2-basic-hardware-transfer.zip` | 13,166,526 B | `626e8bb4a5eb6ccda9bf116fec889dd50bd40a7eaba93f72496137f8588017b3` |
| `gpt2-basic-launch-kit.zip` | 43,466,239 B | `50789f184b0df4c56c095f30e4d9c53f93fbfd5ffce955f3cd7a16aa2afa8762` |
| `preview_release_manifest.md` | 7,862 B | `095733a79b77b10a9b19f7e25cf73df5fa5dc6fc0f228d96af3a94d3a6cd0425` |

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
