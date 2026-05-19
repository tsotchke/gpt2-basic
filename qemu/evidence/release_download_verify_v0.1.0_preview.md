# GPT2-BASIC v0.1.0-preview Clean Download Verification

Verified: `2026-05-19T22:23:09Z`

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
| `gpt2-basic-preview.zip` | 27,854,876 B | `08930e9bbc0b9f0d46ac04059c37c5696b41dcc295e7a06fd9ab75e34d4dbaf3` |
| `gpt2-basic-dosbox.zip` | 13,335,578 B | `4b3ccbbb9c4904d1c02e9a55752d75bdfb5333b1a50d0d5a6b9bfb1b45ddf7b5` |
| `gpt2-basic-hardware-transfer.zip` | 13,328,823 B | `822cee2ba80d3643b6b96065303600c77d7774960badcf7ee4ba8541e3a7273a` |
| `gpt2-basic-launch-kit.zip` | 56,684,708 B | `a78eb7b813f422e077c2f86a0f9cb1fb7a0d3f5eb2495d5efabf16b757b1ffdc` |
| `preview_release_manifest.md` | 8,088 B | `e39a96b72857b148d729b17532cf172a17fe3fa0f2810bc68e5da9435b681506` |

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
