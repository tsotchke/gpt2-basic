# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `PASS`
Average score: `0.878`
Prompt pass rate: `5/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.943 | 4 | 0.0% | 2 | 0 | no | PASS |
| 486_target | 0.740 | 1 | 33.3% | 2 | 0 | no | PASS |
| dos_model | 0.942 | 3 | 0.0% | 2 | 0 | no | PASS |
| basic_runtime | 0.864 | 3 | 28.2% | 2 | 0 | no | PASS |
| optimization | 0.898 | 2 | 15.4% | 2 | 0 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 the model Generation is real only when the DOS executable loads trained weights and computes the next byte through the transformer path. Basic transformer runtime depends on BASIC, arrays, fixed and tokens, loops, model. The useful answer is concrete: it names DOS, weights, arrays, loops, fixed-point math, an
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 s. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. Real inference has two practical details. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. Correctness comes before speed because a fast fake model says nothing about the actual hardware system. Real inference has two practical details. Cor
```

### dos_model

Prompt: `DOS language models need`

```text
 executable, The runtime should stop instead of pretending if the trained model files are absent. Basic transformer runtime depends on BASIC, arrays, fixed and tokens, loops, model. The useful answer is concrete: it names DOS, weights, arrays, loops, fixed-point math, and
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 has ran it. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. For GPT2-BASIC, The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. Correctness comes before speed because a fast fake model says nothing about the actual hardware system. Real inference has two practical details. Correc
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 stwo practical details. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes. Correctness comes before speed because a fast fake model says nothing about the actual hardware system. Real inference has two practical details. Correctness comes before speed because a fast fake model says nothing about the actual hardware system. Vector parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result. The measurable target is better quality without losing speed, me
```

