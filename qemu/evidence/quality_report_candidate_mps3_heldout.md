# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.716`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.690 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_timing | 0.687 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_limits | 0.692 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.821 | 1 | 0.0% | 1 | no | PASS |
| heldout_profiles | 0.690 | 0 | 0.0% | 3 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 loops screendion tatate. It not commacl be ftter runts, attention, sampling, and the trad
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The can ditsk. It tonses stext bytes, look up embeddings, reuse the KV cache, run attenti
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The prompt, explains the machine, and continues with one or two clear sentences instead o
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s of rus, reada trained weights, execute the transformer, and print the continuation witho
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The dre smodel easid a FAS work and pa passsd dista da cohoere text work frokens, weights
```

