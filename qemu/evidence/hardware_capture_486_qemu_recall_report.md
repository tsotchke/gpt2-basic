# Assistant Recall Benchmark

Status: `PASS`
Recall case count: `42`
Average retrieval time: `82 ms`
Max retrieval time: `170 ms`
Average recall score: `42`
Pack counts: `CHAT=12 DEV=6 DOSHELP=9 OFFICE=9 PORTABLE=6`
Recall modes: `kb2_term=42`

This benchmark is generated from `ASSIST.EXE --recall-probe` and measures local pack recall without model generation.

| Pack | Recall | Score | Retrieve ms | Query | Answer |
|---|---|---:|---:|---|---|
| CHAT | kb2_term | 33 | 110 | how can i ask better questions | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | kb2_term | 36 | 110 | what makes this intelligent on a small computer | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | kb2_term | 36 | 50 | which pack should i use for writing | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | kb2_term | 33 | 110 | can this work without the internet | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | kb2_term | 33 | 110 | how do i recover from a bad answer | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | kb2_term | 36 | 60 | what proof helps me trust this | Trust evidence: Trust proof comes from visible files, local weights, reproducible tests, and QEMU or hardware logs. |
| CHAT | kb2_term | 39 | 110 | how should i compare options | Compare options: Name the options, list one tradeoff for each, then choose the practical next step. |
| CHAT | kb2_term | 66 | 110 | help me plan work in small steps | Planning work: Break the job into small steps, do the blocking step first, and verify each result. |
| CHAT | kb2_term | 42 | 110 | what should a useful answer look like | Useful answer: A useful answer should be brief, concrete, honest about limits, and easy to act on. |
| CHAT | kb2_term | 24 | 50 | can you explain something simply | Simple explanation: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | kb2_term | 51 | 110 | what can you know without web access | No web access: Without internet, I cannot fetch news or live facts; use local notes or give the facts in the prompt. |
| CHAT | kb2_term | 36 | 50 | how do i show confidence in an answer | Answer confidence: Say what is known from local files, what is inferred, and what remains uncertain. |
| DOSHELP | kb2_term | 57 | 110 | what happens before autoexec bat runs | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | kb2_term | 24 | 60 | why use 8.3 filenames in batches | DOS filenames: Use 8.3 filenames for maximum DOS compatibility and predictable batch files. |
| DOSHELP | kb2_term | 48 | 110 | how should i prepare files for real hardware | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| DOSHELP | kb2_term | 39 | 50 | what should i do when cwsdpmi is missing | Missing CWSDPMI: If a protected-mode program fails to start, copy CWSDPMI.EXE beside it and rerun the command. |
| DOSHELP | kb2_term | 54 | 110 | how do i mount the dosbox bundle | DOSBox mount: Mount the bundle directory as C:, change to C:\GPT2, then run the batch file for the desired profile. |
| DOSHELP | kb2_term | 60 | 60 | what if the fat image is full | FAT image full: Remove host-only training files or grow the disk image when FAT image assembly runs out of space. |
| DOSHELP | kb2_term | 39 | 110 | what logs matter from qemu | QEMU logs: Capture compile logs, run logs, and copied evidence files before trusting an emulator result. |
| DOSHELP | kb2_term | 57 | 50 | how do i handle a dos memory error | DOS memory error: Free conventional memory by unloading TSRs, loading drivers high, or using a smaller profile. |
| DOSHELP | kb2_term | 36 | 170 | how should a batch menu work | Batch menu: Offer numbered choices, validate the input, and keep each branch short and reversible. |
| OFFICE | kb2_term | 36 | 50 | how should i write a handoff note | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | kb2_term | 36 | 110 | what belongs in a bug report | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | kb2_term | 36 | 60 | make a compact release note | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
| OFFICE | kb2_term | 51 | 50 | what should meeting notes capture | Meeting notes: Capture decisions, owners, dates, open questions, and follow-up actions. |
| OFFICE | kb2_term | 36 | 110 | help me write a project plan | Project plan: List the goal, milestones, owners, risks, and the next checkpoint. |
| OFFICE | kb2_term | 24 | 60 | how do i track risks | Risk register: For each risk, record impact, likelihood, mitigation, owner, and review date. |
| OFFICE | kb2_term | 36 | 50 | what is a useful test plan | Test plan: Define scope, cases, expected results, evidence files, and pass or fail criteria. |
| OFFICE | kb2_term | 36 | 110 | how should i reply to a customer | Customer reply: Acknowledge the issue, give the current status, state the next action, and avoid overpromising. |
| OFFICE | kb2_term | 51 | 60 | how do i write user docs | User docs: Write the task goal, prerequisites, exact steps, expected result, and troubleshooting note. |
| DEV | kb2_term | 36 | 50 | how can this feel modern on a 486 | Modern 486 LLM path: Use small hot-loaded weights, compact retrieval databases, persistent memory, and short synthesis replies. |
| DEV | kb2_term | 36 | 110 | what does retrieval first mean | Retrieval first: Answer from KDB, USER notes, memory, and golden rows before asking the small model to synthesize. |
| DEV | kb2_term | 39 | 60 | how do i author a pack | Pack authoring: Write HELP and KNOW rows, rebuild KDB, run the authoring validator, then run retrieval and QEMU gates. |
| DEV | kb2_term | 42 | 110 | what should i check before release | Release check: Verify tests, logs, artifact names, checksums, release notes, and the target tag. |
| DEV | kb2_term | 45 | 50 | how should we store fast recall data | High velocity recall: Compile notes into compact keyword rows so DOS scans less text and reaches the answer faster. |
| DEV | kb2_term | 39 | 60 | what should a failure record include | Failure record: Record the command, input, expected result, actual result, log path, and next experiment. |
| PORTABLE | kb2_term | 57 | 110 | what does portable intelligence mean | portable meaning: Portable intelligence means small local model weights, retrieval, and memory can run on old machines without a network. |
| PORTABLE | kb2_term | 57 | 50 | why is basic useful for teaching ai | basic teaching: BASIC is useful for teaching machine intelligence because plain arrays, files, and integer arithmetic make the mechanism inspectable. |
| PORTABLE | kb2_term | 15 | 60 | how could this move to c or assembly | runtime ports: The same assistant contract can be reimplemented in C, assembly, Eshkol, or calculator BASIC when files, arrays, and loops exist. |
| PORTABLE | kb2_term | 45 | 50 | why do hot swappable weights matter | domain weight loading: Hot swappable weights load domain behavior into a tiny resident shell without rebuilding the whole runtime. |
| PORTABLE | kb2_term | 72 | 110 | how should tiny machines store recall | tiny machine recall: Tiny machines should store recall as compact indexed rows so slow processors scan fewer bytes before answering. |
| PORTABLE | kb2_term | 63 | 60 | what proof shows this works on old hardware | old hardware proof: Proof for old hardware needs local logs, repeatable tests, QEMU or hardware captures, and visible source files. |
