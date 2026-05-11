# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.701`
Prompt pass rate: `1/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.834 | 1 | 0.0% | 1 | 0 | yes | PASS |
| 486_target | 0.697 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.700 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.822 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| optimization | 0.700 | 0 | 0.0% | 3 | 0 | no | RETRAIN |
| heldout_cache | 0.695 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_timing | 0.623 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_limits | 0.586 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.688 | 0 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_profiles | 0.668 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 hardware the trained is the fundamental algorithmic ackoeporackhet treneradions.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 improcOS tK the comptied comprementation apristed the nexperistialy bed-Picit Poncess , t
```

### dos_model

Prompt: `DOS language models need`

```text
 demonstrate and transformer models tembeded devices - Memorveation thrat cuall achirstect
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 userntal alguid to demonstrate models hardware, doel tansformating ouccum of the from the
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 fundamentally algorithms a trained the fundamental tillly isplemented on on comnstrated t
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 of AI dems divicical dits dicus aseptable bytexps beted-1downcarily vinstation for more i
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 The con the generate inkerad byte from revice.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 Thes. The could boe preational.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 teantuand tin ear. OpnAI Bstututututututur can bestters text from ards smis 2.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The generated oinar he innoveral for modern edge AI development: 1.
```
