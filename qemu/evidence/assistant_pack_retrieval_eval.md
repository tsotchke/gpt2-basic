# Assistant Pack Retrieval Evaluation

Status: `PASS`
Retrieval pass rate: `30/30`

This gate checks local HELP.TXT and KNOW.TXT retrieval for useful paraphrases before model generation.

| Pack | Query | Source | Score | Status | Reason | Answer |
|---|---|---|---:|---|---|---|
| CHAT | how can i ask better questions | KNOW | 123 | PASS |  | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | what makes this intelligent on a small computer | KNOW | 36 | PASS |  | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | which pack should i use for writing | KNOW | 24 | PASS |  | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | can this work without the internet | HELP | 109 | PASS |  | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | how do i recover from a bad answer | KNOW | 21 | PASS |  | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | what proof helps me trust this | HELP | 103 | PASS |  | Trust check: The demo uses local model weights in DOS, plus visible files, tests, and logs. |
| CHAT | how should i compare options | KNOW | 134 | PASS |  | Compare options: Name the options, list one tradeoff for each, then choose the practical next step. |
| CHAT | help me plan work in small steps | KNOW | 131 | PASS |  | Planning work: Break the job into small steps, do the blocking step first, and verify each result. |
| CHAT | what should a useful answer look like | KNOW | 30 | PASS |  | Useful answer: A useful answer should be brief, concrete, honest about limits, and easy to act on. |
| CHAT | can you explain something simply | HELP | 105 | PASS |  | Explain mode: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | what can you know without web access | KNOW | 27 | PASS |  | No web access: Without internet, I cannot fetch news or live facts; use local notes or give the facts in the prompt. |
| CHAT | how do i show confidence in an answer | KNOW | 108 | PASS |  | Confidence signal: Say what is known from local files, what is inferred, and what remains uncertain. |
| DOSHELP | what happens before autoexec bat runs | KNOW | 137 | PASS |  | AUTOEXEC.BAT tuning: CONFIG.SYS loads drivers before AUTOEXEC.BAT; then AUTOEXEC sets PATH, variables, and required TSR tools. |
| DOSHELP | why use 8.3 filenames in batches | HELP | 88 | PASS |  | Batch file help: In a batch file, use IF EXIST checks, clear status messages, and 8.3 DOS-compatible names for model files. |
| DOSHELP | how should i prepare files for real hardware | KNOW | 24 | PASS |  | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| DOSHELP | what should i do when cwsdpmi is missing | HELP | 87 | PASS |  | DPMI host: Protected-mode DOS programs need a DPMI host such as CWSDPMI.EXE beside the program. |
| DOSHELP | how do i mount the dosbox bundle | KNOW | 42 | PASS |  | DOSBox mount: Mount the bundle directory as C:, change to C:\GPT2, then run the batch file for the desired profile. |
| DOSHELP | what if the fat image is full | KNOW | 60 | PASS |  | FAT image full: Remove host-only training files or grow the disk image when FAT image assembly runs out of space. |
| DOSHELP | what logs matter from qemu | KNOW | 39 | PASS |  | QEMU logs: Capture compile logs, run logs, and copied evidence files before trusting an emulator result. |
| DOSHELP | how do i handle a dos memory error | KNOW | 137 | PASS |  | DOS memory error: Free conventional memory by unloading TSRs, loading drivers high, or using a smaller profile. |
| DOSHELP | how should a batch menu work | KNOW | 126 | PASS |  | Batch menu: Offer numbered choices, validate the input, and keep each branch short and reversible. |
| OFFICE | how should i write a handoff note | KNOW | 128 | PASS |  | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | what belongs in a bug report | KNOW | 126 | PASS |  | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | make a compact release note | KNOW | 128 | PASS |  | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
| OFFICE | what should meeting notes capture | KNOW | 132 | PASS |  | Meeting notes: Capture decisions, owners, dates, open questions, and follow-up actions. |
| OFFICE | help me write a project plan | KNOW | 128 | PASS |  | Project plan: List the goal, milestones, owners, risks, and the next checkpoint. |
| OFFICE | how do i track risks | KNOW | 115 | PASS |  | Risk register: For each risk, record impact, likelihood, mitigation, owner, and review date. |
| OFFICE | what is a useful test plan | KNOW | 125 | PASS |  | Test plan: Define scope, cases, expected results, evidence files, and pass or fail criteria. |
| OFFICE | how should i reply to a customer | KNOW | 36 | PASS |  | Customer reply: Acknowledge the issue, give the current status, state the next action, and avoid overpromising. |
| OFFICE | how do i write user docs | KNOW | 128 | PASS |  | User docs: Write the task goal, prerequisites, exact steps, expected result, and troubleshooting note. |
