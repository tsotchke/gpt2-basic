# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.691`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.696 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_timing | 0.623 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_limits | 0.694 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.774 | 1 | 0.0% | 3 | yes | PASS |
| heldout_profiles | 0.668 | 0 | 0.0% | 2 | yes | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
al of modern AI developmenter sundgrive sus that mick odep for the read sinstruce to be tr
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The con the generate inkerad byte from revice.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 Thes. The could boe preational. It should notalgic apesed beford the develop forward syst
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s the core text from logits, corrroun plocupumes or on at 486.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The generated oinar he innoveral for modern edge AI development: 1.
```

