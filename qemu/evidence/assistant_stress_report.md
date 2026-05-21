# Assistant Stress Report

Status: `PASS`

Reply count: `44`
Source counts: `golden=26 retrieval=10 model=0 fallback=0 memory=8`
Average total reply time: `132 ms`
Average retrieval time: `81 ms`
Recall modes: `kb2_bucket=3 kb2_term=40 none=1`

| Pack | Source | Recall | Total ms | Query | Answer |
|---|---|---|---:|---|---|
| CHAT | golden | kb2_term | 160 | why did my answer repeat itself | If I repeat, reset the prompt and ask one shorter question. |
| CHAT | retrieval | kb2_term | 220 | tell me why this old computer model matters | Old hardware: The demo matters because a tiny local model can run on old DOS-style hardware without a network. |
| CHAT | golden | kb2_term | 170 | make a tiny plan for fixing a bug | Check the first error, change one thing, then test again. |
| CHAT | retrieval | kb2_term | 220 | what is the difference between a prompt and an answer | Explain mode: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | golden | kb2_term | 0 | can you explain what local inference means | Local inference means the DOS program reads model weights and produces the answer on this machine. |
| CHAT | golden | kb2_bucket | 270 | i feel stuck debugging this | Check the first error, change one thing, then test again. |
| CHAT | golden | kb2_term | 170 | what should i do if the answer sounds weird | Retry with a shorter prompt or switch packs. |
| CHAT | retrieval | kb2_term | 160 | give me a status update about a delayed release | Release status: Check the tag target, release assets, checksums, and test result before calling the release done. |
| CHAT | golden | kb2_term | 160 | can you browse the internet from dos | I cannot browse the internet from DOS. |
| CHAT | golden | kb2_term | 0 | can we talk about games | Yes, games are a fine topic. |
| CHAT | golden | none | 380 | i am tired | Rest if you can. |
| CHAT | golden | kb2_bucket | 160 | i feel lonely | I can keep you company briefly. |
| CHAT | retrieval | kb2_term | 220 | do you enjoy music | Music topic: I can talk about music. |
| CHAT | golden | kb2_term | 0 | what should i do if i am bored | Try one small project. |
| CHAT | golden | kb2_term | 170 | how do i relax for a minute | Breathe slowly and rest for a minute. |
| CHAT | golden | kb2_term | 110 | what is friendship | Friendship is care and trust. |
| CHAT | golden | kb2_term | 170 | what can we discuss | We can discuss ideas, feelings, games, or DOS. |
| CHAT | golden | kb2_term | 110 | what is your favorite food | I do not eat, but I can talk about food. |
| CHAT | golden | kb2_term | 110 | what is a goal | A goal is something you want to reach. |
| CHAT | golden | kb2_term | 170 | how do i improve | Practice one small thing each day. |
| CHAT | memory | kb2_term | 0 | my name is Operator | I will remember your name is Operator. |
| CHAT | memory | kb2_term | 170 | what is my name | Your name is Operator. |
| CHAT | memory | kb2_term | 160 | we are working on the DOSBox assistant | I will remember we are working on the DOSBox assistant. |
| CHAT | memory | kb2_term | 170 | what are we working on | We are working on the DOSBox assistant. |
| CHAT | memory | kb2_term | 160 | i prefer short answers | I will remember you prefer short answers. |
| CHAT | memory | kb2_term | 170 | how should you answer me | I should answer short answers. |
| CHAT | memory | kb2_term | 0 | what did i just ask | You just asked: how should you answer me. |
| CHAT | memory | kb2_term | 110 | what do you remember | Memory: name=Operator; goal=the DOSBox assistant; style=short answers; last_user=what did i just ask. |
| DOSHELP | retrieval | kb2_term | 0 | how do i keep conventional memory free | Conventional memory: Keep drivers high, trim TSR programs, and preserve low memory for old DOS programs. |
| DOSHELP | retrieval | kb2_term | 170 | my autoexec is too long what should i change | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | golden | kb2_term | 110 | how should i clean autoexec.bat | Keep AUTOEXEC BAT short, trim PATH, and load resident tools only when needed. |
| DOSHELP | retrieval | kb2_term | 160 | write a batch command that checks for model files | Batch file help: In a batch file, use IF EXIST checks, clear status messages, and 8.3 DOS-compatible names for model files. |
| DOSHELP | retrieval | kb2_term | 170 | why does protected mode need a dpmi host | DPMI host: Protected-mode DOS programs need a DPMI host such as CWSDPMI.EXE beside the program. |
| DOSHELP | retrieval | kb2_term | 110 | what does config.sys do | CONFIG.SYS example: Use HIMEM, DOS HIGH UMB, FILES 30, and BUFFERS 20 as the safe CONFIG SYS baseline. |
| OFFICE | golden | kb2_term | 170 | make this sentence sound professional: the release broke | Use direct, polite, professional wording, name the release problem, and end with the next action. |
| OFFICE | golden | kb2_term | 110 | summarize this: tests passed but the tag was stale | Summary: tests passed, the tag was stale, and the next action is to refresh the tag. |
| OFFICE | golden | kb2_bucket | 0 | summarize: tests passed but dosbox needed a helper file | Summary: tests passed, DOSBox needed a helper file. Include it. |
| OFFICE | golden | kb2_term | 110 | shorten: we need to verify the release before publishing | Short version: verify the release before publishing. |
| OFFICE | golden | kb2_term | 170 | write a polite status update about a delayed build | Use direct, polite, concrete wording, include the current blocker, and end with the next action. |
| OFFICE | golden | kb2_term | 110 | make this clearer: the artifact uploaded but the tag was stale | State that the artifact uploaded, the tag was stale, and the next action is to refresh the tag. |
| DEV | golden | kb2_term | 110 | how can this feel modern on a 486 | Use small hot-loaded weights, compact retrieval databases, persistent memory, and short synthesis replies. |
| DEV | retrieval | kb2_term | 0 | what does retrieval first mean | Retrieval first: Answer from KDB, USER notes, memory, and golden rows before asking the small model to synthesize. |
| DEV | golden | kb2_term | 160 | how do i author a pack | Write HELP and KNOW rows, rebuild KDB, run the validator, then run retrieval and QEMU gates. |
| DEV | golden | kb2_term | 110 | what should i check before release | Verify tests, logs, artifact names, checksums, release notes, and the target tag. |
