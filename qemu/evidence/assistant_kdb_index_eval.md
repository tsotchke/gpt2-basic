# Assistant KDB Index Evaluation

Status: `PASS`
Indexed recall pass rate: `42/42`
Candidate rows scanned: `858/1617`
Candidate scan ratio: `0.531`

This gate mirrors the DOS bucket fast path before the full KDB fallback.

| Pack | Query | Buckets | Rows | Status | Reason | Answer |
|---|---|---|---:|---|---|---|
| CHAT | how can i ask better questions | A,B,Q | 29/78 | PASS |  | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | what makes this intelligent on a small computer | M,I,S,C | 64/78 | PASS |  | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | which pack should i use for writing | W,P | 28/78 | PASS |  | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | can this work without the internet | W,I | 21/78 | PASS |  | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | how do i recover from a bad answer | R,B,A | 35/78 | PASS |  | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | what proof helps me trust this | P,H,T | 52/78 | PASS |  | Trust evidence: Trust proof comes from visible files, local weights, reproducible tests, and QEMU or hardware logs. |
| CHAT | how should i compare options | C,O | 39/78 | PASS |  | Compare options: Name the options, list one tradeoff for each, then choose the practical next step. |
| CHAT | help me plan work in small steps | H,P,W,S | 54/78 | PASS |  | Planning work: Break the job into small steps, do the blocking step first, and verify each result. |
| CHAT | what should a useful answer look like | U,A,L | 43/78 | PASS |  | Useful answer: A useful answer should be brief, concrete, honest about limits, and easy to act on. |
| CHAT | can you explain something simply | E,S | 29/78 | PASS |  | Simple explanation: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | what can you know without web access | K,W,A | 29/78 | PASS |  | No web access: Without internet, I cannot fetch news or live facts; use local notes or give the facts in the prompt. |
| CHAT | how do i show confidence in an answer | S,C,A | 52/78 | PASS |  | Answer confidence: Say what is known from local files, what is inferred, and what remains uncertain. |
| DOSHELP | what happens before autoexec bat runs | H,B,A,R | 19/26 | PASS |  | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | why use 8.3 filenames in batches | F,B | 12/26 | PASS |  | DOS filenames: Use 8.3 filenames for maximum DOS compatibility and predictable batch files. |
| DOSHELP | how should i prepare files for real hardware | P,F,R,H | 18/26 | PASS |  | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| DOSHELP | what should i do when cwsdpmi is missing | C,M | 19/26 | PASS |  | Missing CWSDPMI: If a protected-mode program fails to start, copy CWSDPMI.EXE beside it and rerun the command. |
| DOSHELP | how do i mount the dosbox bundle | M,D,B | 20/26 | PASS |  | DOSBox mount: Mount the bundle directory as C:, change to C:\GPT2, then run the batch file for the desired profile. |
| DOSHELP | what if the fat image is full | F,I | 6/26 | PASS |  | FAT image full: Remove host-only training files or grow the disk image when FAT image assembly runs out of space. |
| DOSHELP | what logs matter from qemu | L,M,Q | 12/26 | PASS |  | QEMU logs: Capture compile logs, run logs, and copied evidence files before trusting an emulator result. |
| DOSHELP | how do i handle a dos memory error | H,D,M,E | 23/26 | PASS |  | DOS memory error: Free conventional memory by unloading TSRs, loading drivers high, or using a smaller profile. |
| DOSHELP | how should a batch menu work | B,M,W | 14/26 | PASS |  | Batch menu: Offer numbered choices, validate the input, and keep each branch short and reversible. |
| OFFICE | how should i write a handoff note | W,H,N | 7/27 | PASS |  | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | what belongs in a bug report | B,R | 10/27 | PASS |  | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | make a compact release note | M,C,R,N | 19/27 | PASS |  | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
| OFFICE | what should meeting notes capture | M,N,C | 14/27 | PASS |  | Meeting notes: Capture decisions, owners, dates, open questions, and follow-up actions. |
| OFFICE | help me write a project plan | H,W,P | 12/27 | PASS |  | Project plan: List the goal, milestones, owners, risks, and the next checkpoint. |
| OFFICE | how do i track risks | T,R | 13/27 | PASS |  | Risk register: For each risk, record impact, likelihood, mitigation, owner, and review date. |
| OFFICE | what is a useful test plan | U,T,P | 12/27 | PASS |  | Test plan: Define scope, cases, expected results, evidence files, and pass or fail criteria. |
| OFFICE | how should i reply to a customer | R,C | 15/27 | PASS |  | Customer reply: Acknowledge the issue, give the current status, state the next action, and avoid overpromising. |
| OFFICE | how do i write user docs | W,U,D | 12/27 | PASS |  | User docs: Write the task goal, prerequisites, exact steps, expected result, and troubleshooting note. |
| DEV | how can this feel modern on a 486 | F,M,4 | 10/23 | PASS |  | Modern 486 LLM path: Use small hot-loaded weights, compact retrieval databases, persistent memory, and short synthesis replies. |
| DEV | what does retrieval first mean | D,R,F,M | 16/23 | PASS |  | Retrieval first: Answer from KDB, USER notes, memory, and golden rows before asking the small model to synthesize. |
| DEV | how do i author a pack | A,P | 12/23 | PASS |  | Pack authoring: Write HELP and KNOW rows, rebuild KDB, run the authoring validator, then run retrieval and QEMU gates. |
| DEV | what should i check before release | C,B,R | 14/23 | PASS |  | Release check: Verify tests, logs, artifact names, checksums, release notes, and the target tag. |
| DEV | how should we store fast recall data | S,F,R,D | 16/23 | PASS |  | High velocity recall: Compile notes into compact keyword rows so DOS scans less text and reaches the answer faster. |
| DEV | what should a failure record include | F,R,I | 12/23 | PASS |  | Failure record: Record the command, input, expected result, actual result, log path, and next experiment. |
| PORTABLE | what does portable intelligence mean | D,P,I,M | 11/11 | PASS |  | portable meaning: Portable intelligence means small local model weights, retrieval, and memory can run on old machines without a network. |
| PORTABLE | why is basic useful for teaching ai | B,U,T | 4/11 | PASS |  | basic teaching: BASIC is useful for teaching machine intelligence because plain arrays, files, and integer arithmetic make the mechanism inspectable. |
| PORTABLE | how could this move to c or assembly | M,A | 8/11 | PASS |  | runtime ports: The same assistant contract can be reimplemented in C, assembly, Eshkol, or calculator BASIC when files, arrays, and loops exist. |
| PORTABLE | why do hot swappable weights matter | H,S,W,M | 8/11 | PASS |  | domain weight loading: Hot swappable weights load domain behavior into a tiny resident shell without rebuilding the whole runtime. |
| PORTABLE | how should tiny machines store recall | T,M,S,R | 6/11 | PASS |  | tiny machine recall: Tiny machines should store recall as compact indexed rows so slow processors scan fewer bytes before answering. |
| PORTABLE | what proof shows this works on old hardware | P,S,W,O,H | 9/11 | PASS |  | old hardware proof: Proof for old hardware needs local logs, repeatable tests, QEMU or hardware captures, and visible source files. |
