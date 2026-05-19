# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.739`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.601 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| 486_target | 0.819 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| dos_model | 0.818 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| basic_runtime | 0.654 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| optimization | 0.698 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.657 | 0 | 0.0% | 4 | 0 | yes | RETRAIN |
| heldout_timing | 0.688 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_limits | 0.815 | 1 | 0.0% | 3 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.946 | 2 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_profiles | 0.696 | 0 | 0.0% | 3 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 That The refub, thy., Howory cos CPC, RFIN.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 compatible contrat memory, neds tokens, pragrame large emory, and smataxt by the atand the sm thokn per
```

### dos_model

Prompt: `DOS language models need`

```text
 should stepl math; ardings, reuse repeorted sis the one f radeden weights, execut stord by produce cach saled s
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 projects logits, a she cloud truntime work from the ale path.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 he trained weights, transformer should show that is charaintl produced the complexit, the project devion specti
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 before singlllle suay suabld hardware clacere with runtime.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Woull th to 486 tran text byte comes from the trained weights, transformer layers, logits, and sample a ste
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 4 486 486docommps It cabenit minstan miplatical sulgity within looop hardware logs iterates than and opera
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 arithmetic cost, and fixed point math the model production eded to tokens, explainic storight abic nvalual memory
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 the that thestiming through: 1. Trans reated should say that The runtime strippported iscoduction comes fro
```
