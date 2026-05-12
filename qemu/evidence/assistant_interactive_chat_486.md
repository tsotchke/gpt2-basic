# GPT2-BASIC Interactive CHAT 486 Evidence

Date: 2026-05-12

Command:

```sh
QEMU_DISPLAY=curses bash qemu/run_assistant_interactive_486.sh
```

This run used the real interactive entrypoint. It did not pass `--scripted`,
did not redirect `ASSIST.EXE` output to a log file, and accepted typed prompts
at the DOS `>` prompt.

Observed sequence:

- FreeDOS booted and `ASSIST.EXE` compiled successfully.
- `ASSIST.EXE` opened in `CHAT` with `Model: PACKS\CHAT\MODEL`.
- Before the first `>` prompt, the shell printed
  `Loading CHAT model before prompt...`, loaded the 4096-token CHAT vocabulary,
  and then printed `Loaded model: PACKS\CHAT\MODEL (486dx2-usable)`.
- Prompt `i am bored` did not reload the model. It streamed visible
  `Thinking:` progress including prompt token count, `ctx` token pieces, and
  output-token sampling before returning `Answer: Try one small project.`
- `/u` opened the in-DOS transcript view and showed the user prompt plus
  assistant reply; `/d` returned toward the latest page.
- Earlier manual prompt `how are you` loaded `PACKS\CHAT\MODEL`, printed
  `Thinking:` progress, and returned `Answer: I answer in DOS.`
- Prompt `hello` reused the loaded CHAT model, printed `Thinking: .....`, and
  returned `Answer: Hello from DOS.`
- `/quit` exited the assistant back to the DOS prompt.

This confirms the product demo path can be talked to directly and that CHAT
answers are produced by the fixed-point model inside DOS rather than by the
retrieval-only scripted evidence path. As of the preload fix, the first typed
question no longer pays the model-load cost; startup and `/pack NAME` perform
the active-pack load before returning to the prompt. The visible `Thinking:`
stream is inference progress, not a scripted answer: it exposes prompt/context
token prefill and output-token sampling while keeping the final answer stream
separate.
