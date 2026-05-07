# GPT2-BASIC Domain Curriculum

This corpus is project-owned training material generated from the actual production target: DOS fixed-point GPT2-BASIC inference. It is intended for fine-tuning after permissive online pretraining, not for replacing runtime evidence.

## Summary

- Corpus: `data/domain_curriculum/domain_curriculum.txt`
- Documents: 760
- Clean characters: 384725
- Seed: 486
- Variants per topic: 48
- Cross-topic variants: 160
- Prompt-family variants: 36
- Anchor completion repeats: 0
- Answer-bank repeats: 18
- Exact held-out prompt leakage: no

## Topic Matrix

| Topic | Keywords |
|---|---|
| `cache` | cache, context, tokens, speed, memory, reuse |
| `timing` | timing, seconds, tokens, measured, hardware, QEMU |
| `limits` | memory, context, speed, weights, small, hardware |
| `fixed_point` | fixed, integer, weights, arithmetic, logits, runtime |
| `profiles` | profile, quality, speed, memory, measure, tradeoff |
| `runtime` | BASIC, arrays, fixed, tokens, loops, model |
| `real_inference` | logits, trained, weights, transformer, runtime |

## Fine-Tune Command

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --init-model-dir assets/gpt2_basic/MODEL --include-docs --corpus-file data/domain_curriculum/domain_curriculum.txt --corpus-weight 2 --device mps --steps 1000 --output assets/gpt2_basic/MODEL_DOMAIN_CANDIDATE
```

Promote only if host held-out quality beats the active DOS baseline, then run DOS vector parity, DOS held-out quality, and QEMU `--perf`.
