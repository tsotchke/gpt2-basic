# Assistant Pack Retrieval Evaluation

Status: `PASS`
Retrieval pass rate: `12/12`

This gate checks local HELP.TXT and KNOW.TXT retrieval for useful paraphrases before model generation.

| Pack | Query | Source | Score | Status | Reason | Answer |
|---|---|---|---:|---|---|---|
| CHAT | how can i ask better questions | KNOW | 123 | PASS |  | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | what makes this intelligent on a small computer | KNOW | 36 | PASS |  | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | which pack should i use for writing | KNOW | 24 | PASS |  | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | can this work without the internet | HELP | 109 | PASS |  | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | how do i recover from a bad answer | KNOW | 21 | PASS |  | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | what proof helps me trust this | HELP | 103 | PASS |  | Trust check: The demo uses local model weights in DOS, plus visible files, tests, and logs. |
| DOSHELP | what happens before autoexec bat runs | HELP | 121 | PASS |  | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | why use 8.3 filenames in batches | HELP | 88 | PASS |  | Batch file help: In a batch file, use IF EXIST checks, clear status messages, and 8.3 DOS-compatible names for model files. |
| DOSHELP | how should i prepare files for real hardware | KNOW | 24 | PASS |  | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| OFFICE | how should i write a handoff note | KNOW | 128 | PASS |  | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | what belongs in a bug report | KNOW | 126 | PASS |  | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | make a compact release note | KNOW | 128 | PASS |  | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
