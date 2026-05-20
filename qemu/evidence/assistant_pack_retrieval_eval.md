# Assistant Pack Retrieval Evaluation

Status: `PASS`
Retrieval pass rate: `36/36`

This gate checks local KDB retrieval for useful paraphrases before model generation.

| Pack | Query | Source | Score | Status | Reason | Answer |
|---|---|---|---:|---|---|---|
| CHAT | how can i ask better questions | KDB | 33 | PASS |  | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | what makes this intelligent on a small computer | KDB | 36 | PASS |  | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | which pack should i use for writing | KDB | 36 | PASS |  | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | can this work without the internet | KDB | 33 | PASS |  | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | how do i recover from a bad answer | KDB | 33 | PASS |  | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | what proof helps me trust this | KDB | 36 | PASS |  | Trust evidence: Trust proof comes from visible files, local weights, reproducible tests, and QEMU or hardware logs. |
| CHAT | how should i compare options | KDB | 39 | PASS |  | Compare options: Name the options, list one tradeoff for each, then choose the practical next step. |
| CHAT | help me plan work in small steps | KDB | 66 | PASS |  | Planning work: Break the job into small steps, do the blocking step first, and verify each result. |
| CHAT | what should a useful answer look like | KDB | 42 | PASS |  | Useful answer: A useful answer should be brief, concrete, honest about limits, and easy to act on. |
| CHAT | can you explain something simply | KDB | 24 | PASS |  | Simple explanation: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | what can you know without web access | KDB | 51 | PASS |  | No web access: Without internet, I cannot fetch news or live facts; use local notes or give the facts in the prompt. |
| CHAT | how do i show confidence in an answer | KDB | 36 | PASS |  | Answer confidence: Say what is known from local files, what is inferred, and what remains uncertain. |
| DOSHELP | what happens before autoexec bat runs | KDB | 57 | PASS |  | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | why use 8.3 filenames in batches | KDB | 24 | PASS |  | DOS filenames: Use 8.3 filenames for maximum DOS compatibility and predictable batch files. |
| DOSHELP | how should i prepare files for real hardware | KDB | 48 | PASS |  | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| DOSHELP | what should i do when cwsdpmi is missing | KDB | 39 | PASS |  | Missing CWSDPMI: If a protected-mode program fails to start, copy CWSDPMI.EXE beside it and rerun the command. |
| DOSHELP | how do i mount the dosbox bundle | KDB | 54 | PASS |  | DOSBox mount: Mount the bundle directory as C:, change to C:\GPT2, then run the batch file for the desired profile. |
| DOSHELP | what if the fat image is full | KDB | 60 | PASS |  | FAT image full: Remove host-only training files or grow the disk image when FAT image assembly runs out of space. |
| DOSHELP | what logs matter from qemu | KDB | 39 | PASS |  | QEMU logs: Capture compile logs, run logs, and copied evidence files before trusting an emulator result. |
| DOSHELP | how do i handle a dos memory error | KDB | 57 | PASS |  | DOS memory error: Free conventional memory by unloading TSRs, loading drivers high, or using a smaller profile. |
| DOSHELP | how should a batch menu work | KDB | 36 | PASS |  | Batch menu: Offer numbered choices, validate the input, and keep each branch short and reversible. |
| OFFICE | how should i write a handoff note | KDB | 36 | PASS |  | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | what belongs in a bug report | KDB | 36 | PASS |  | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | make a compact release note | KDB | 36 | PASS |  | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
| OFFICE | what should meeting notes capture | KDB | 51 | PASS |  | Meeting notes: Capture decisions, owners, dates, open questions, and follow-up actions. |
| OFFICE | help me write a project plan | KDB | 36 | PASS |  | Project plan: List the goal, milestones, owners, risks, and the next checkpoint. |
| OFFICE | how do i track risks | KDB | 24 | PASS |  | Risk register: For each risk, record impact, likelihood, mitigation, owner, and review date. |
| OFFICE | what is a useful test plan | KDB | 36 | PASS |  | Test plan: Define scope, cases, expected results, evidence files, and pass or fail criteria. |
| OFFICE | how should i reply to a customer | KDB | 36 | PASS |  | Customer reply: Acknowledge the issue, give the current status, state the next action, and avoid overpromising. |
| OFFICE | how do i write user docs | KDB | 51 | PASS |  | User docs: Write the task goal, prerequisites, exact steps, expected result, and troubleshooting note. |
| DEV | how can this feel modern on a 486 | KDB | 36 | PASS |  | Modern 486 LLM path: Use small hot-loaded weights, compact retrieval databases, persistent memory, and short synthesis replies. |
| DEV | what does retrieval first mean | KDB | 36 | PASS |  | Retrieval first: Answer from KDB, USER notes, memory, and golden rows before asking the small model to synthesize. |
| DEV | how do i author a pack | KDB | 39 | PASS |  | Pack authoring: Write HELP and KNOW rows, rebuild KDB, run the authoring validator, then run retrieval and QEMU gates. |
| DEV | what should i check before release | KDB | 42 | PASS |  | Release check: Verify tests, logs, artifact names, checksums, release notes, and the target tag. |
| DEV | how should we store fast recall data | KDB | 45 | PASS |  | High velocity recall: Compile notes into compact keyword rows so DOS scans less text and reaches the answer faster. |
| DEV | what should a failure record include | KDB | 39 | PASS |  | Failure record: Record the command, input, expected result, actual result, log path, and next experiment. |
