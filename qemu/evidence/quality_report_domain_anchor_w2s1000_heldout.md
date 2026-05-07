# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.752`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.942 | 2 | 0.0% | 2 | no | PASS |
| heldout_timing | 0.689 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_limits | 0.628 | 0 | 0.0% | 5 | yes | RETRAIN |
| heldout_fixed_point | 0.812 | 1 | 0.0% | 1 | no | PASS |
| heldout_profiles | 0.690 | 0 | 0.0% | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache deperiated revices - EEMU utt proior construct relss and memory, run and and decode
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
m, MAIGPT in a 486-era promerft rained revices lang reaporates model for modern edge AI de
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The INGnes T2XPesct, Landix.EEEEE proace. The target.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 on of optimples. The answer should be short, concrete, and tied to the DOS fixed-point ru
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
m BASIC cache ontens an transformer models are and edge devices - SresurpD : Concluse and 
```

