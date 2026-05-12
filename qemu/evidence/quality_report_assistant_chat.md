# Assistant Pack Quality Report: CHAT

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab478`
Evaluation backend: `fixed`
Quality suite: `assistant-pack`
Quality status: `PASS`
Average score: `1.000`
Prompt pass rate: `4/4` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| chat_hello | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how are you | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_what can you do | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_limits | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |

## Generated Continuations

### chat_hello

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: Hello. Note: Starting a conversation: Hello from DOS. Assistant:`

```text
Hello from DOS.
```

### chat_how are you

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: How are you? Note: Status check: I answer in DOS. Assistant:`

```text
I answer in DOS.
```

### chat_what can you do

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: Hello, what can you do? Note: Assistant abilities: I chat in DOS. Assistant:`

```text
I chat in DOS.
```

### chat_limits

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: What are your limits? Note: Model limits: I am tiny and local. Assistant:`

```text
I am tiny and local.
```

