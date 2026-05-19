# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.700`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.708 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_timing | 0.697 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.704 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_profiles | 0.695 | 0 | 0.0% | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 begins. The answer should be short, concrete, and tied to the DOS fixed-point runtime.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 C: The implementation surpeses and model princeXEMUndating and this: model strained weigh
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 This program ays abut constrained on embeddings, causal self attention lo enoughid in the
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s suseful model answer with predictable inferencius the clear imithically implementation a
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 C DOS transformer opt, model insteger arithmetic with the relevant words metatching his c
```

