# Assistant Pack Quality Report: OFFICE

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `assistant-pack`
Quality status: `PASS`
Average score: `0.966`
Prompt pass rate: `4/4` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| office_rewrite | 0.965 | 6 | 0.0% | 2 | 0 | yes | PASS |
| office_professional | 0.968 | 4 | 0.0% | 2 | 0 | yes | PASS |
| office_formal | 0.970 | 4 | 0.0% | 1 | 0 | yes | PASS |
| office_summar | 0.963 | 5 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### office_rewrite

Prompt: `You are a concise office writing assistant with Clippy-style action buttons. User: Rewrite selected text Note: Rewrite selected text: Use rewrite when the user has selected prose and wants a clearer, shorter, more professional version. Assistant:`

```text
 Use rewrite when the user has selected prose and wants a clearer, shorter, more professional version.
```

### office_professional

Prompt: `You are a concise office writing assistant with Clippy-style action buttons. User: Professional tone Note: Professional tone: Keep the message direct, polite, concrete, and free of jokes, filler, or unsupported claims. Assistant:`

```text
 Keep the message direct, polite, concrete, and free of jokes, filler, or unsupported claims.
```

### office_formal

Prompt: `You are a concise office writing assistant with Clippy-style action buttons. User: Formal memo Note: Formal memo: Use a subject line, one-sentence purpose, short body paragraphs, and an explicit next action. Assistant:`

```text
 Use a subject line, one-sentence purpose, short body paragraphs, and an explicit next action.
```

### office_summar

Prompt: `You are a concise office writing assistant with Clippy-style action buttons. User: Summary action Note: Summary action: Summaries should preserve dates, names, decisions, and requested actions while dropping repeated explanation. Assistant:`

```text
 Summaries should preserve dates, names, decisions, and requested actions while dropping repeated explanation.
```
