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
- Prompt `how are you` loaded `PACKS\CHAT\MODEL`, printed `Thinking:` progress,
  and returned `Answer: I answer in DOS.`
- Prompt `hello` reused the loaded CHAT model, printed `Thinking: .....`, and
  returned `Answer: Hello from DOS.`
- `/quit` exited the assistant back to the DOS prompt.

This confirms the product demo path can be talked to directly and that CHAT
answers are produced by the fixed-point model inside DOS rather than by the
retrieval-only scripted evidence path.
