# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.776`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.817 | 1 | 0.0% | 2 | no | PASS |
| 486_target | 0.646 | 0 | 0.0% | 1 | yes | RETRAIN |
| dos_model | 0.823 | 1 | 0.0% | 2 | no | PASS |
| basic_runtime | 0.945 | 2 | 0.0% | 2 | no | PASS |
| optimization | 0.652 | 0 | 0.0% | 3 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The BASIC runtime uses by the run on old entworks, and embedd GPUntraing the model should
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 proabile.bas te the exp omplerar. On the dusentage model.
```

### dos_model

Prompt: `DOS language models need`

```text
s a small hing into the prompt about comullat, the prompt prior is concrespeded developmen
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses arrays for tokens, embeddings, runs econs that these fundamento matchinud oden trans
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 the answer should mention a 486. EM. --- Dunstimang reming report.
```

