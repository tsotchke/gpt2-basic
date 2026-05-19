# Assistant Stress Report

Status: `PASS`

Reply count: `18`
Source counts: `golden=1 retrieval=15 model=0 fallback=2`

| Pack | Source | Query | Answer |
|---|---|---|---|
| CHAT | retrieval | why did my answer repeat itself | Repeat control: If I repeat, ask a shorter question or switch packs. |
| CHAT | retrieval | tell me why this old computer model matters | Old hardware: The demo matters because a tiny local model can run on old DOS-style hardware without a network. |
| CHAT | retrieval | make a tiny plan for fixing a bug | Debug plan: Start with the failing command, expected result, and first error line, then test one small fix. |
| CHAT | retrieval | what is the difference between a prompt and an answer | Prompt meaning: A prompt is your typed question. |
| CHAT | golden | can you explain what local inference means | Local inference means the DOS program reads model weights and produces the answer on this machine. |
| CHAT | retrieval | i feel stuck debugging this | Debug plan: Start with the failing command, expected result, and first error line, then test one small fix. |
| CHAT | retrieval | what should i do if the answer sounds weird | Retry advice: Ask a shorter question, switch packs if needed, and treat strange output as a signal to retry. |
| CHAT | retrieval | give me a status update about a delayed release | Release status: Check the tag target, release assets, checksums, and test result before calling the release done. |
| DOSHELP | retrieval | how do i keep conventional memory free | CONFIG.SYS memory: Load HIMEM.SYS first, use DOS=HIGH,UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | retrieval | my autoexec is too long what should i change | AUTOEXEC.BAT hygiene: Keep PATH short, load resident tools only when needed, and make sure the assistant can find PACKS and MODEL. |
| DOSHELP | retrieval | write a batch command that checks for model files | Batch file help: Use IF EXIST checks, clear status messages, and 8-dot-3 names for DOS, Windows DOS boxes, and OS/2 sessions. |
| DOSHELP | retrieval | why does protected mode need a dpmi host | DPMI host: Protected-mode DOS programs need a DPMI host such as CWSDPMI.EXE beside the program. |
| DOSHELP | retrieval | what does config.sys do | CONFIG.SYS example: Use DEVICE=HIMEM.SYS, DOS=HIGH,UMB, FILES=30, and BUFFERS=20 as the safe baseline. |
| OFFICE | retrieval | make this sentence sound professional: the release broke | Professional tone: Keep the message direct, polite, concrete, and free of jokes, filler, or unsupported claims. |
| OFFICE | retrieval | summarize: tests passed but dosbox needed a helper file | Summary action: Summaries should preserve dates, names, decisions, and requested actions while dropping repeated explanation. |
| OFFICE | retrieval | shorten: we need to verify the release before publishing | Shorten action: Shortening should keep the original intent and remove qualifiers, duplicate phrases, and low-value background. |
| OFFICE | fallback | write a polite status update about a delayed build | Use direct, polite wording, keep the concrete fact, and end with the next action. |
| OFFICE | fallback | make this clearer: the artifact uploaded but the tag was stale | State what happened, why it matters, and the next action in one short paragraph. |
