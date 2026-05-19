# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab258`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.711`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.700 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_timing | 0.696 | 0 | 0.0% | 1 | no | RETRAIN |
| heldout_limits | 0.815 | 1 | 0.0% | 1 | no | PASS |
| heldout_fixed_point | 0.650 | 0 | 0.0% | 2 | yes | RETRAIN |
| heldout_profiles | 0.695 | 0 | 0.0% | 2 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 with both insptimization and and ssecrial fully optimizations production bes tecontin and
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
If user n a 486 proablit comput afil modernation profile usefunctal operatined on expontin
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 MOS decode displing memory dise not be large to futut computing contexer spect, and then 
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
arse the and of randon log prate embedd bete bytes tokens.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The user oun or that transformer arctument ad a sunder should hoopse fixed-point inferenc
```

