# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.769`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.688 | 0 | 0.0% | 1 | no | RETRAIN |
| 486_target | 0.819 | 1 | 0.0% | 1 | no | PASS |
| dos_model | 0.819 | 1 | 0.0% | 1 | no | PASS |
| basic_runtime | 0.822 | 1 | 0.0% | 1 | no | PASS |
| optimization | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 That The refub, thy., Howory cos CPC, RFIN., Paraiks, C context fixed-point inference continue code the context w
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 compatible contrat memory, neds tokens, pragrame large emory, and smataxt by the atand the sm thokn per
```

### dos_model

Prompt: `DOS language models need`

```text
 should stepl math; ardings, reuse repeorted sis the one f radeden weights, executiong stand the sating syster what the tr
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
tict cans should be target whthat checkpoint computer be fixed-size les: kearing weights, transformer layers, lo
```

### optimization

Prompt: `To improve performance on real hardware`

```text
, QEMU. A QEMU result raing weights neeration nostrad, and production point, make attention cache choste
```

