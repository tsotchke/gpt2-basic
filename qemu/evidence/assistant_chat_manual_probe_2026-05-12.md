# Assistant CHAT Manual QEMU Probe: 2026-05-12

Runtime path:

- `QEMU_DISPLAY=curses bash qemu/run_assistant_interactive_486.sh`
- FreeDOS 486 profile, `ASSIST.EXE` compiled inside the VM
- CHAT pack model: `PACKS\CHAT\MODEL`
- CHAT model profile: `486dx2-usable`
- CHAT tokenizer: lexicon, `vocab=4096`
- CHAT fixed quality report: `PASS`, `25/25`
- Interactive preload check: `Loading CHAT model before prompt...` appeared
  before the first `>` prompt; the first typed question did not print
  `Loading model...`.
- Thinking stream check: the DOS console printed prompt token count, `ctx`
  token pieces, and output-token sampling before the answer text.
- Transcript check: `/u` displayed the in-DOS transcript without relying on
  QEMU terminal scrollback.

Typed prompts and observed DOS console replies:

| Prompt | Observed reply |
| --- | --- |
| `i am bored` | `Try one small project.` |
| `tell me a joke` | `DOS smiled because it found its prompt.` |
| `do you like music` | `I can talk about music.` |
| `what is a prompt` | `A prompt is your typed question.` |
| `what is promp` | `A prompt is your typed question.` |

The casual replies were checked after expanding the CHAT pack to a 4096-token
lexicon vocabulary and retraining on broader common English call-and-response
dialogue. The prompt-definition replies were checked after narrowing
`AssistCleanGeneratedText` so ordinary words such as `prompt` are no longer
mistaken for leaked prompt labels. The scripted QEMU evidence still covers the
other packs and structured `ASSIST_MODEL` records. The DOS console showed
`Thinking:` progress followed by streamed `Answer:` text from the loaded CHAT
model. The current interactive path preloads the active pack model during
startup and after `/pack NAME`, before returning control to the prompt. The
current transcript path supports short `/u`, `/d`, and `/h` commands for paging
inside DOS.
