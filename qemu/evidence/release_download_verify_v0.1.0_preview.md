# GPT2-BASIC v0.1.0-preview Clean Download Verification

Verified: `2026-05-18T04:58:00Z`

Repository: `Tsotchke-Corporation/gpt2-basic`

Release: `https://github.com/Tsotchke-Corporation/gpt2-basic/releases/tag/v0.1.0-preview`

Issue: `https://github.com/Tsotchke-Corporation/gpt2-basic/issues/2`

## Download Source

The assets were downloaded from GitHub release `v0.1.0-preview` into a fresh
temporary directory, not reused from the local release build output:

```text
/private/tmp/gpt2-basic-release-download.XfIaxr
```

Downloaded files:

```text
gpt2-basic-preview.zip
gpt2-basic-preview.zip.sha256
gpt2-basic-hardware-transfer.zip
gpt2-basic-hardware-transfer.zip.sha256
preview_release_manifest.md
```

## Checksums

The local `shasum` command failed because of a host Perl locale panic, so the
same SHA-256 checks were performed with `openssl dgst -sha256`.

| Asset | Size | SHA-256 |
|---|---:|---|
| `gpt2-basic-preview.zip` | 27,400,459 B | `b6ae31b82dcef92cc8be48f4162c9c85c53765b388bd48366cb3e6cb45fabfa1` |
| `gpt2-basic-hardware-transfer.zip` | 12,766,204 B | `59bac4e53222a5513dc907bfa142d9b46a8db209e4d1c7c2ede8dc0309e9e55c` |
| `gpt2-basic-preview.zip.sha256` | 89 B | `3a7e82f22c20808381fb10ba9a8c2c1e691ad9e8a85deb9c3b52483ba3224c62` |
| `gpt2-basic-hardware-transfer.zip.sha256` | 99 B | `a7487c218d736f8111e09eeab64e754e25370d1398411e40c66a2081c64c2225` |
| `preview_release_manifest.md` | 7,381 B | `60d513040507f0315ca2f06017dbbdb4ad02a7cc3a0936507079f06259c7e458` |

The zip hashes match the contents of the downloaded `.sha256` sidecars.

## Extraction

Both zips were extracted into a separate fresh temporary root:

```text
/private/tmp/gpt2-basic-release-extract.lHeiUW
```

The extracted hardware bundle included:

```text
/private/tmp/gpt2-basic-release-extract.lHeiUW/gpt2-basic-hardware-transfer/MANIFEST.TXT
```

## Verifier Command

The verifier was run from the extracted preview tree with explicit paths to the
downloaded zips, sidecars, extracted hardware bundle, and downloaded manifest:

```sh
python3 scripts/verify_preview_artifacts.py \
  --preview-dir /private/tmp/gpt2-basic-release-extract.lHeiUW/gpt2-basic-preview \
  --preview-zip /private/tmp/gpt2-basic-release-download.XfIaxr/gpt2-basic-preview.zip \
  --preview-zip-sha256 /private/tmp/gpt2-basic-release-download.XfIaxr/gpt2-basic-preview.zip.sha256 \
  --hardware-dir /private/tmp/gpt2-basic-release-extract.lHeiUW/gpt2-basic-hardware-transfer \
  --hardware-zip /private/tmp/gpt2-basic-release-download.XfIaxr/gpt2-basic-hardware-transfer.zip \
  --hardware-zip-sha256 /private/tmp/gpt2-basic-release-download.XfIaxr/gpt2-basic-hardware-transfer.zip.sha256 \
  --repo-manifest /private/tmp/gpt2-basic-release-download.XfIaxr/preview_release_manifest.md
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

This closes the consumer-style download verification gate for
`v0.1.0-preview`.
