# Assistant KDB Term Index Evaluation

Status: `PASS`
Term-index recall pass rate: `42/42`
Candidate rows scored: `113/1617`
Candidate row ratio: `0.070`
Term-index plus record bytes touched: `71125/675360`
Candidate byte ratio: `0.105`

This gate mirrors the DOS strongest KB2T?.TXT shard fast path before KB2TERM.TXT and KB2 bucket fallback.

| Pack | Query | Terms | Index Files | Rows | Bytes | Status | Reason | Answer |
|---|---|---|---|---:|---:|---|---|---|
| CHAT | how can i ask better questions | ask,better,questions | KB2TQ.TXT | 2/78 | 1002/32512 | PASS |  | Better prompts: Say the goal, give one detail, and ask for the next useful step. |
| CHAT | what makes this intelligent on a small computer | makes,intelligent,small,computer | KB2TI.TXT,KB2TERM.TXT | 14/78 | 10350/32512 | PASS |  | Small-computer usefulness: A tiny local model becomes more useful with retrieval, memory, and quick focused help without a network. |
| CHAT | which pack should i use for writing | which,pack,writing | KB2TW.TXT | 1/78 | 678/32512 | PASS |  | Pack switching: Use CHAT for conversation, DOSHELP for DOS setup, and OFFICE for writing tasks. |
| CHAT | can this work without the internet | work,without,internet | KB2TI.TXT | 2/78 | 1065/32512 | PASS |  | Network limit: I cannot browse the internet from DOS; I answer from local model weights and pack files. |
| CHAT | how do i recover from a bad answer | recover,bad,answer | KB2TR.TXT | 2/78 | 1197/32512 | PASS |  | Mistake recovery: If an answer is wrong, ask a shorter question, switch packs, or give the exact error. |
| CHAT | what proof helps me trust this | proof,helps,trust | KB2TT.TXT | 3/78 | 1685/32512 | PASS |  | Trust evidence: Trust proof comes from visible files, local weights, reproducible tests, and QEMU or hardware logs. |
| CHAT | how should i compare options | compare,options | KB2TO.TXT | 1/78 | 677/32512 | PASS |  | Compare options: Name the options, list one tradeoff for each, then choose the practical next step. |
| CHAT | help me plan work in small steps | help,plan,work,small,steps | KB2TS.TXT | 13/78 | 5899/32512 | PASS |  | Planning work: Break the job into small steps, do the blocking step first, and verify each result. |
| CHAT | what should a useful answer look like | useful,answer,look,like | KB2TU.TXT | 8/78 | 3516/32512 | PASS |  | Useful answer: A useful answer should be brief, concrete, honest about limits, and easy to act on. |
| CHAT | can you explain something simply | explain,something,simply | KB2TS.TXT | 2/78 | 1323/32512 | PASS |  | Simple explanation: Use plain words, one example, and a short answer that fits the prompt. |
| CHAT | what can you know without web access | know,without,web,access | KB2TW.TXT | 3/78 | 1510/32512 | PASS |  | No web access: Without internet, I cannot fetch news or live facts; use local notes or give the facts in the prompt. |
| CHAT | how do i show confidence in an answer | show,confidence,answer | KB2TC.TXT | 1/78 | 888/32512 | PASS |  | Answer confidence: Say what is known from local files, what is inferred, and what remains uncertain. |
| DOSHELP | what happens before autoexec bat runs | happens,before,autoexec,bat,runs | KB2TA.TXT | 2/26 | 988/10880 | PASS |  | AUTOEXEC.BAT hygiene: CONFIG.SYS loads drivers first, then AUTOEXEC.BAT runs commands; keep PATH short and trim resident tools. |
| DOSHELP | why use 8.3 filenames in batches | filenames,batches | KB2TF.TXT | 1/26 | 627/10880 | PASS |  | DOS filenames: Use 8.3 filenames for maximum DOS compatibility and predictable batch files. |
| DOSHELP | how should i prepare files for real hardware | prepare,files,real,hardware | KB2TH.TXT | 4/26 | 1865/10880 | PASS |  | Hardware copy: Copy GPT2, MODEL, PACKS, CWSDPMI, and batch files together before testing on real DOS. |
| DOSHELP | what should i do when cwsdpmi is missing | cwsdpmi,missing | KB2TM.TXT | 1/26 | 615/10880 | PASS |  | Missing CWSDPMI: If a protected-mode program fails to start, copy CWSDPMI.EXE beside it and rerun the command. |
| DOSHELP | how do i mount the dosbox bundle | mount,dosbox,bundle | KB2TD.TXT | 2/26 | 1080/10880 | PASS |  | DOSBox mount: Mount the bundle directory as C:, change to C:\GPT2, then run the batch file for the desired profile. |
| DOSHELP | what if the fat image is full | fat,image,full | KB2TI.TXT | 1/26 | 532/10880 | PASS |  | FAT image full: Remove host-only training files or grow the disk image when FAT image assembly runs out of space. |
| DOSHELP | what logs matter from qemu | logs,matter,qemu | KB2TM.TXT,KB2TERM.TXT | 3/26 | 3640/10880 | PASS |  | QEMU logs: Capture compile logs, run logs, and copied evidence files before trusting an emulator result. |
| DOSHELP | how do i handle a dos memory error | handle,dos,memory,error | KB2TM.TXT | 3/26 | 1447/10880 | PASS |  | DOS memory error: Free conventional memory by unloading TSRs, loading drivers high, or using a smaller profile. |
| DOSHELP | how should a batch menu work | batch,menu,work | KB2TB.TXT | 7/26 | 3161/10880 | PASS |  | Batch menu: Offer numbered choices, validate the input, and keep each branch short and reversible. |
| OFFICE | how should i write a handoff note | write,handoff,note | KB2TH.TXT | 1/27 | 553/11296 | PASS |  | Handoff note: Say what is done, what remains, where evidence lives, and who owns the next action. |
| OFFICE | what belongs in a bug report | belongs,bug,report | KB2TB.TXT | 1/27 | 585/11296 | PASS |  | Bug report shape: Include expected behavior, actual behavior, reproduction steps, logs, and the suspected area. |
| OFFICE | make a compact release note | make,compact,release,note | KB2TR.TXT | 1/27 | 745/11296 | PASS |  | Release note shape: Lead with what changed, list proof, then state any known limits plainly. |
| OFFICE | what should meeting notes capture | meeting,notes,capture | KB2TM.TXT | 1/27 | 598/11296 | PASS |  | Meeting notes: Capture decisions, owners, dates, open questions, and follow-up actions. |
| OFFICE | help me write a project plan | help,write,project,plan | KB2TP.TXT | 2/27 | 1090/11296 | PASS |  | Project plan: List the goal, milestones, owners, risks, and the next checkpoint. |
| OFFICE | how do i track risks | track,risks | KB2TT.TXT | 1/27 | 628/11296 | PASS |  | Risk register: For each risk, record impact, likelihood, mitigation, owner, and review date. |
| OFFICE | what is a useful test plan | useful,test,plan | KB2TU.TXT,KB2TERM.TXT | 3/27 | 3855/11296 | PASS |  | Test plan: Define scope, cases, expected results, evidence files, and pass or fail criteria. |
| OFFICE | how should i reply to a customer | reply,customer | KB2TC.TXT | 1/27 | 801/11296 | PASS |  | Customer reply: Acknowledge the issue, give the current status, state the next action, and avoid overpromising. |
| OFFICE | how do i write user docs | write,user,docs | KB2TW.TXT | 1/27 | 567/11296 | PASS |  | User docs: Write the task goal, prerequisites, exact steps, expected result, and troubleshooting note. |
| DEV | how can this feel modern on a 486 | feel,modern,486 | KB2TM.TXT | 1/23 | 617/9632 | PASS |  | Modern 486 LLM path: Use small hot-loaded weights, compact retrieval databases, persistent memory, and short synthesis replies. |
| DEV | what does retrieval first mean | does,retrieval,first,mean | KB2TR.TXT | 4/23 | 2041/9632 | PASS |  | Retrieval first: Answer from KDB, USER notes, memory, and golden rows before asking the small model to synthesize. |
| DEV | how do i author a pack | author,pack | KB2TA.TXT,KB2TERM.TXT | 3/23 | 3891/9632 | PASS |  | Pack authoring: Write HELP and KNOW rows, rebuild KDB, run the authoring validator, then run retrieval and QEMU gates. |
| DEV | what should i check before release | check,before,release | KB2TR.TXT | 1/23 | 793/9632 | PASS |  | Release check: Verify tests, logs, artifact names, checksums, release notes, and the target tag. |
| DEV | how should we store fast recall data | store,fast,recall,data | KB2TR.TXT | 1/23 | 793/9632 | PASS |  | High velocity recall: Compile notes into compact keyword rows so DOS scans less text and reaches the answer faster. |
| DEV | what should a failure record include | failure,record,include | KB2TI.TXT,KB2TERM.TXT | 4/23 | 4185/9632 | PASS |  | Failure record: Record the command, input, expected result, actual result, log path, and next experiment. |
| PORTABLE | what does portable intelligence mean | does,portable,intelligence,mean | KB2TI.TXT | 5/11 | 2250/4640 | PASS |  | portable meaning: Portable intelligence means small local model weights, retrieval, and memory can run on old machines without a network. |
| PORTABLE | why is basic useful for teaching ai | basic,useful,teaching | KB2TT.TXT | 1/11 | 553/4640 | PASS |  | basic teaching: BASIC is useful for teaching machine intelligence because plain arrays, files, and integer arithmetic make the mechanism inspectable. |
| PORTABLE | how could this move to c or assembly | move,assembly | KB2TA.TXT | 1/11 | 625/4640 | PASS |  | runtime ports: The same assistant contract can be reimplemented in C, assembly, Eshkol, or calculator BASIC when files, arrays, and loops exist. |
| PORTABLE | why do hot swappable weights matter | hot,swappable,weights,matter | KB2TS.TXT | 1/11 | 632/4640 | PASS |  | domain weight loading: Hot swappable weights load domain behavior into a tiny resident shell without rebuilding the whole runtime. |
| PORTABLE | how should tiny machines store recall | tiny,machines,store,recall | KB2TM.TXT | 2/11 | 1047/4640 | PASS |  | tiny machine recall: Tiny machines should store recall as compact indexed rows so slow processors scan fewer bytes before answering. |
| PORTABLE | what proof shows this works on old hardware | proof,shows,works,old,hardware | KB2TH.TXT | 1/11 | 531/4640 | PASS |  | old hardware proof: Proof for old hardware needs local logs, repeatable tests, QEMU or hardware captures, and visible source files. |
