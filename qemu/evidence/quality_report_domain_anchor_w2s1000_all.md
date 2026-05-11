# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.729`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.817 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| 486_target | 0.592 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| dos_model | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.945 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| optimization | 0.576 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_cache | 0.942 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_timing | 0.687 | 0 | 0.0% | 3 | 0 | no | RETRAIN |
| heldout_limits | 0.603 | 0 | 0.0% | 5 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.812 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.624 | 0 | 0.0% | 3 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC runtime uses by the run on old entworks, and embedd GPUntraing the model should
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 proabile.bas te the exp omplerar.
```

### dos_model

Prompt: `DOS language models need`

```text
 eustationag check constraints makes the ternig report generitt work peinton, embeddings,
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses arrays for tokens, embeddings, runs econs that these fundamento matchinud oden trans
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 the answer should mention a 486.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache deperiated revices - EEMU utt proior construct relss and memory, run and and decode
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 This usthis milll usimple on DOS, a 486 progrvsed neds be that can tichef omparing report
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The INGnes T2XPesct, Landix.EEEEE proace.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 on of optimples. The answer should be short, concrete, and tied to the DOS fixed-point ru
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The BASIC BASIC runtime uses orrrpected cons.
```
