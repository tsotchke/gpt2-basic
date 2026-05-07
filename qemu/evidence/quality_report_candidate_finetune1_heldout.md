# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.763`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.807 | 1 | 0.0% | 1 | no | PASS |
| heldout_timing | 0.692 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_limits | 0.685 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.944 | 2 | 0.0% | 2 | no | PASS |
| heldout_profiles | 0.689 | 0 | 0.0% | 2 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
al from a prompt. Me. MOn aske and comple. Ites leay, , trunding the trad context window, 
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Bemoryse. A slansure, predictable data formats, and the satisfaction of running from a no
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The user should boot DOS, run GPT2.EXE, enter a prompt, and see the trained model continu
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
d bytext-forward ivectors the vectors, logits, and fixed point loops so the DOS program ca
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The Yes. Ondince the model is trained on the host, exported as small binary weights, copi
```

