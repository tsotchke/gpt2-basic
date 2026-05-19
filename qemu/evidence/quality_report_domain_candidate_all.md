# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.687`
Prompt pass rate: `0/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.689 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| 486_target | 0.680 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| dos_model | 0.696 | 0 | 0.0% | 3 | 0 | no | RETRAIN |
| basic_runtime | 0.624 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| optimization | 0.692 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_cache | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_timing | 0.683 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| heldout_limits | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.594 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| heldout_profiles | 0.822 | 1 | 0.0% | 1 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The Contig innovation ofts, ecorrte meain profile, quality, speed, and then tie them to m
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 dems, preducing the ker(6 sult, and sample profile arrly: pemons, compute-2.
```

### dos_model

Prompt: `DOS language models need`

```text
 weight file the context window deeht releval output logisorrr inciplity-Frooity-vectures
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 with the result should repofile gones arther.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 system. The tadetes stoker only inference with timing, seconds, tokens, measured, hardwar
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 loop. SitReal beenserationsmbing bufficid-eoninty, and Pattion, and non cloingue Constr s
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Cruntime uses a compath both DOS runs efuly answer onlay thardward pass.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The The uservisted bysing understand tokens make attentention continue the text from the
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 of optible loopsions the need: GPT2.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The uservis with in the measurable tach themor infersteails operations by memory, context
```
