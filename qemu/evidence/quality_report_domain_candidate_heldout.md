# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.741`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_timing | 0.683 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.809 | 1 | 0.0% | 2 | no | PASS |
| heldout_profiles | 0.822 | 1 | 0.0% | 1 | no | PASS |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 loop. SitReal beenserationsmbing bufficid-eoninty, and Pattion, and non cloingue Constr s
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Cruntime uses a compath both DOS runs efuly answer onlay thardward pass.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The The uservisted bysing understand tokens make attentention continue the text from the 
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 of optible loopsions the need: GPT2.EXE --perf 486DEDEL qemu/evidence/bes fixed-point log
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The uservis with in the measurable tach themor infersteails operations by memory, context
```

