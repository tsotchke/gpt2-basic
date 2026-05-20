# Assistant Raw Prompt Evaluation

Status: `PASS`
Prompt pass rate: `26/26`

| Pack | Query | Status | Reason | Completion |
|---|---|---|---|---|
| CHAT | why are you saying the same phrase again | PASS |  | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | explain local inference without jargon | PASS |  | Local inference means the model runs on this machine. The answer comes from local model weights. |
| CHAT | is this answer coming from real model weights | PASS |  | Yes, the answer comes from local model weights. Type a question and I will answer. |
| CHAT | what should i do if the output looks wrong | PASS |  | Retry with a shorter prompt or switch packs. |
| CHAT | give me a three step bug fixing plan | PASS |  | Check the first error, change one thing, change one thing, then test again. |
| CHAT | what is the difference between my prompt and your answer | PASS |  | A prompt is what you type. An answer is the model output. That is the simple version. |
| CHAT | why does an old computer demo matter | PASS |  | Old hardware proves the local model is small and practical. |
| CHAT | tell me whether the release is ready | PASS |  | Check the tag, assets, checksums, and test result. |
| CHAT | i am stuck and need a small next step | PASS |  | Choose one small step, test it, and then adjust. Check the first error, change one thing, then test again. |
| CHAT | can you make a tiny plan for today | PASS |  | Pick one goal, list three steps, then start small. Ask one clear prompt and I will keep going. |
| CHAT | tell me a short story about a dos model | PASS |  | A tiny model woke up inside DOS and answered one prompt. It runs locally in this DOS demo. |
| CHAT | what are your limits in this session | PASS |  | I am tiny and local. This tiny session has clear limits. |
| CHAT | can you browse the internet from dos | PASS |  | I cannot browse from DOS. It runs locally in this DOS demo. |
| CHAT | what does a token mean | PASS |  | A token is a small piece of text. |
| CHAT | how do i focus for a few minutes | PASS |  | Remove one distraction and choose one small task. |
| CHAT | write one friendly sentence about this demo | PASS |  | The demo is live and runs locally in DOS. Type a question and I will answer. |
| DOSHELP | why does my protected mode program need cwsdpmi | PASS |  | Protected-mode DOS programs need a DPMI host such as CWSDPMI beside the program. |
| DOSHELP | how do i leave more conventional memory free | PASS |  | Load HIMEM first, use DOS HIGH UMB, keep FILES and BUFFERS modest, and preserve conventional memory. |
| DOSHELP | what should i put in config.sys for this assistant | PASS |  | Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| DOSHELP | write a safe batch check for the model directory | PASS |  | Use IF EXIST checks, clear status messages, and 8-dot-3 names for DOS, Windows DOS boxes, and OS/2 sessions. |
| DOSHELP | my autoexec is messy and slow | PASS |  | Keep PATH short, load resident tools only when needed, and make sure the assistant can find PACKS and MODEL. |
| OFFICE | rewrite this politely: the artifact failed | PASS |  | Use direct, polite wording, keep the artifact fact, and end with the next action. |
| OFFICE | summarize this: tests passed but the tag was stale | PASS |  | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | make this clearer: checksums changed after rebuild | PASS |  | State that checksums changed after rebuild, why it matters, and the next action. |
| OFFICE | shorten this sentence without losing the decision | PASS |  | Keep the decision and next action, then remove repeated explanation. |
| OFFICE | write a status update about a delayed build | PASS |  | Use direct polite wording, include the current blocker, and end with the next action or timing. |
