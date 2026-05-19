# GPT2-BASIC Architecture Profile Sweep

This report tracks trainer architecture profiles, not same-shape checkpoint variants.
Rows without an exported model are planning rows until `train_gpt2_basic.py` writes the checkpoint and DOS/QEMU evidence is collected.

## Profile Matrix

| Profile | Export | Shape | Params | Runtime MB | 386DX tok/s | 486SX tok/s | 486DX2 tok/s | Pentium 133 tok/s | Held-out quality | Score |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `386-min` | `assets/gpt2_basic/MODEL_PROFILE_386_MIN` | `2L 32D 4H ctx128 h128 v258` | 46338 | 0.247 | 1.20 | 2.39 | 11.70 measured | 28.71 | NEEDS_TRAINING 0/5 avg 0.579 (dos-fixed-qemu) | 27.42 |
| `486sx-safe` | `assets/gpt2_basic/MODEL` | `2L 48D 4H ctx192 h192 v258` | 90882 | 0.497 | 0.68 | 1.36 | 6.01 measured | 16.27 | NEEDS_TRAINING 0/5 avg 0.685 (dos-fixed-qemu) | 8.29 |
| `486dx2-usable` | `assets/gpt2_basic/MODEL_PROFILE_486DX2_USABLE` | `3L 64D 4H ctx192 h256 v258` | 195650 | 1.038 | 0.30 | 0.59 | 2.65 measured | 7.12 | NEEDS_TRAINING 0/5 avg 0.673 (dos-fixed-qemu) | 1.72 |
| `486dx4-plus` | `missing` | `4L 64D 4H ctx256 h256 v258` | 249730 | 1.463 | 0.23 | 0.45 | 1.81 model | 5.42 | missing | 0.00 |
| `pentium-best` | `missing` | `4L 96D 6H ctx256 h384 v258` | 521922 | 2.755 | 0.11 | 0.22 | 0.89 model | 2.67 | missing | 0.00 |

## Current Finding

3 profiles now have DOS fixed-point held-out evidence. `386-min` is the fastest measured QEMU `486dx2-66` profile at 11.70 tok/s, while `486sx-safe` has the best DOS held-out average at 0.685. No profile passes the held-out suite yet.

## Next Commands

Train missing profile checkpoints, then run the same DOS evidence contract for each exported model directory.

```sh
python3 scripts/architecture_profile_sweep.py --train --profiles 486dx4-plus --steps-override 1200 --include-docs --corpus-file data/online_corpus/online_training_corpus.txt
python3 scripts/architecture_profile_sweep.py --refresh-heldout-float
bash qemu/run_vectors_486.sh assets/gpt2_basic/MODEL_PROFILE_486DX4_PLUS
bash qemu/run_quality_486.sh assets/gpt2_basic/MODEL_PROFILE_486DX4_PLUS
bash qemu/run_perf_486.sh 486dx2-66 assets/gpt2_basic/MODEL_PROFILE_486DX4_PLUS
```

Evidence root: `<repo>/qemu/evidence`
