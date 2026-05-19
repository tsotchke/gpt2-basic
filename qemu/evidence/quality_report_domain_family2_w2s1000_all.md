# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.752`
Prompt pass rate: `1/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.690 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| 486_target | 0.942 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.694 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.873 | 4 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.822 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_cache | 0.711 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_timing | 0.817 | 1 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.660 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.616 | 0 | 0.0% | 1 | 1 | yes | RETRAIN |
| heldout_profiles | 0.694 | 0 | 0.0% | 2 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC useful continuation core instead of drifting into generic prose.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 peress reaieng the trained model files are aveis fully opsent the dat DOS memory, memory,
```

### dos_model

Prompt: `DOS language models need`

```text
 evelory tokens, loops, model. The model should explain its the runtime contuation demonst
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses fixed aroche dat bete tokens, loops, model.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 evidence. The measurable target is better quality without losing speed, memory discipline
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 vidence depends on this: The real model profile comparison an 486s deminst argethe det.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The user should constraints for tokens, model profile, quality, fixed-point phen arials f
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 T2 BASIC, arrays, fixed, lear tradeoff between model mitrations.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 : The smodel is the project det a printable byte.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The uses under should connect finclue with the relevant words are a bud trixed withoins t
```
