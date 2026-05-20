# Assistant Generalist Prompt Evaluation

Status: `PASS`
Prompt pass rate: `24/24`

These prompts exercise CHAT categories such as troubleshooting, trust, memory, offline limits, release checks, and old-hardware context.

| Query | Status | Reason | Completion |
|---|---|---|---|
| guide me through troubleshooting a failure | PASS |  | Check the first error, change one thing, then test again. |
| what first step should i take when debugging fails | PASS |  | Check the first error, change one thing, then test again. |
| i am overwhelmed and need one next task | PASS |  | Choose one small task, then take the first step. |
| my todo list is too big | PASS |  | Choose one small task, then take the first step. |
| why should someone believe this is real | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| how can i tell this is not fake | PASS |  | The demo uses local model weights in DOS. Type a question and I will answer. |
| what if my question is too long | PASS |  | Short prompts work better in this DOS demo. |
| should i type a paragraph or a short prompt | PASS |  | Short prompts work better in this DOS demo. |
| can you talk about history a little | PASS |  | Ask one simple question about the topic. Type a question and I will answer. |
| help me learn a new topic | PASS |  | Ask one simple question about the topic. |
| write a kind line about vintage dos hardware | PASS |  | This old DOS computer can run a friendly local model. Check the tag, assets, checksums, and test result. |
| say something friendly about old dos hardware | PASS |  | This old DOS computer can run a friendly local model. |
| what does local mean in this project | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| explain local in this demo | PASS |  | Local means the model runs on this machine. Type a question and I will answer. |
| do you keep memory during the session | PASS |  | I remember only small facts during this session. |
| can you remember small facts here | PASS |  | I remember only small facts during this session. Ask one clear prompt and I will keep going. |
| how can i stop looped output | PASS |  | Use a shorter prompt and reset if the answer loops. |
| what should i do if output loops | PASS |  | Use a shorter prompt and reset if the answer loops. |
| give me a short release checklist | PASS |  | Check the tag, assets, checksums, and tests. |
| what is a small release checklist | PASS |  | Check the tag, assets, checksums, and tests. |
| explain emulator simply | PASS |  | An emulator runs one machine inside another. |
| what does an emulator do | PASS |  | An emulator runs one machine inside another. |
| why can this not go online | PASS |  | DOS cannot browse here; the answer comes from local files. |
| why is this dos chat offline | PASS |  | DOS cannot browse here; the answer comes from local files. |
