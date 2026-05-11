# Assistant Pack Quality Report: DOSHELP

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `assistant-pack`
Quality status: `PASS`
Average score: `0.954`
Prompt pass rate: `4/4` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| doshelp_memory | 0.970 | 5 | 0.0% | 2 | 0 | yes | PASS |
| doshelp_config_sys | 0.932 | 4 | 0.0% | 2 | 0 | yes | PASS |
| doshelp_autoexec | 0.963 | 3 | 0.0% | 2 | 0 | yes | PASS |
| doshelp_batch | 0.953 | 4 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### doshelp_memory

Prompt: `You are a concise DOS and 486 troubleshooting assistant. User: CONFIG.SYS memory Note: CONFIG.SYS memory: Load HIMEM.SYS first, use DOS=HIGH,UMB, keep FILES and BUFFERS modest, and preserve conventional memory. Assistant:`

```text
 Load HIMEM.SYS first, use DOS=HIGH,UMB, keep FILES and BUFFERS modest, and preserve conventional memory.
```

### doshelp_config_sys

Prompt: `You are a concise DOS and 486 troubleshooting assistant. User: CONFIG.SYS example Note: CONFIG.SYS example: Use DEVICE=HIMEM.SYS, DOS=HIGH,UMB, FILES=30, and BUFFERS=20 as the safe baseline. Assistant:`

```text
 Use DEVICE=HIMEM.SYS, DOS=HIGH,UMB, FILES=30, and BUFFERS=20 as the safe baseline.
```

### doshelp_autoexec

Prompt: `You are a concise DOS and 486 troubleshooting assistant. User: AUTOEXEC.BAT hygiene Note: AUTOEXEC.BAT hygiene: Keep PATH short, load resident tools only when needed, and make sure the assistant can find PACKS and MODEL. Assistant:`

```text
 Keep PATH short, load resident tools only when needed, and make sure the assistant can find PACKS and MODEL.
```

### doshelp_batch

Prompt: `You are a concise DOS and 486 troubleshooting assistant. User: Batch file help Note: Batch file help: Use IF EXIST checks, clear status messages, and 8-dot-3 names for DOS, Windows DOS boxes, and OS/2 sessions. Assistant:`

```text
 Use IF EXIST checks, clear status messages, and 8-dot-3 names for DOS, Windows DOS boxes, and OS/2 sessions.
```
