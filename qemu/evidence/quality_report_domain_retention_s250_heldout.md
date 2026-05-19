# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.745`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.817 | 1 | 0.0% | 2 | yes | PASS |
| heldout_timing | 0.825 | 1 | 0.0% | 2 | yes | PASS |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_fixed_point | 0.694 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_profiles | 0.694 | 0 | 0.0% | 1 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 vidence dependitations models are actuallu iteadables memory for the disk.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
IC C.?X.1.xt--icunt C: C: GPT2.EXP.Berfore.DEXE C: qemu/evidence/profile pareto report.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 C, J. Wullge Dult poweplact ims.batical bits must for reduce recomputation sperformarmert
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 on of lodskens archical hardware evidence depends on this: The code eperations by the mod
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
m The MODOS, arnd production proctxt.md GPT2-BASIC can the transformer semory models are u
```

