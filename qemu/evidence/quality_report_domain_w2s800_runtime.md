# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.819`
Prompt pass rate: `4/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.940 | 2 | 0.0% | 2 | no | PASS |
| 486_target | 0.845 | 1 | 0.0% | 1 | yes | PASS |
| dos_model | 0.819 | 1 | 0.0% | 1 | no | PASS |
| basic_runtime | 0.673 | 0 | 0.0% | 1 | yes | RETRAIN |
| optimization | 0.820 | 1 | 0.0% | 1 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, and sample b
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
-ousencal denout model should talk about files, cach eanswer should say trained weights.
```

### dos_model

Prompt: `DOS language models need`

```text
s a real hardware sks about model in DOS program thical inference, contining, seconds, and
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 run on a 486. A ushould asestion and cacaume chontes an measured output.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile comparison profile usefuncta/d.man trincewer is normpt athe prompt ductical rule 
```

