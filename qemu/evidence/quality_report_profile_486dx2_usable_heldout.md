# GPT2-BASIC Quality Report

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.684`
Prompt pass rate: `0/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.689 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_timing | 0.640 | 0 | 0.0% | 1 | yes | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_fixed_point | 0.698 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.697 | 0 | 0.0% | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 wat whith the corece acturable.b, eard earns on the model computer on from the ding the t
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The quesestions tures that the core from contraints.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The gas as trained a transformers in an rinit emsight is that transformer models computin
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
s instead disings, and the trad2 penticisision with Frectur thaing model computing couth t
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The quantantization and forwardinationation. izers operations of ncomputed GPT-2 BASIC on
```

