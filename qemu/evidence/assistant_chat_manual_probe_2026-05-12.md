# Assistant CHAT Manual QEMU Probe: 2026-05-12

Runtime path:

- `QEMU_DISPLAY=curses bash qemu/run_assistant_interactive_486.sh`
- FreeDOS 486 profile, `ASSIST.EXE` compiled inside the VM
- CHAT pack model: `PACKS\CHAT\MODEL`
- CHAT model profile: `486dx2-usable`
- CHAT tokenizer: lexicon, `vocab=724`

Typed prompts and observed DOS console replies:

| Prompt | Observed reply |
| --- | --- |
| `what is a prompt` | `A prompt is your typed question.` |
| `what is promp` | `A prompt is your typed question.` |
| `is this scripted` | `No, it is real.` |

The prompt-definition replies were checked after narrowing
`AssistCleanGeneratedText` so ordinary words such as `prompt` are no longer
mistaken for leaked prompt labels. The DOS console showed `Thinking:` progress
followed by streamed `Answer:` text from the loaded CHAT model.
