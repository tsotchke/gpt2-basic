# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality status: `NEEDS_TRAINING`
Average score: `0.744`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.834 | 1 | 0.0% | 1 | yes | PASS |
| 486_target | 0.697 | 0 | 0.0% | 2 | no | RETRAIN |
| dos_model | 0.700 | 0 | 0.0% | 2 | no | RETRAIN |
| basic_runtime | 0.789 | 1 | 0.0% | 1 | yes | PASS |
| optimization | 0.700 | 0 | 0.0% | 3 | no | RETRAIN |

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
. I the rougid a sthat trained model is the careful optimization.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 fundamentally algorithms a trained the fundamental tillly isplemented on on comnstrated t
```

