# GPT2-BASIC Assistant Packs

Packs are selected inside `ASSIST.EXE` with `/pack NAME`. Type `/packs` to
list them and `/about` to read the current pack's `USAGE.TXT` instructions
inside DOS.

## CHAT

Use `CHAT` for ordinary conversation. It is first in `PACKS.TXT`, so the
interactive QEMU demo starts here. It uses `PACKS\CHAT\MODEL` and short
conversation notes from `CHAT\HELP.TXT`. The DOS shell prints `Thinking:`
progress with visible prompt/context token pieces and output-token sampling,
then streams the generated `Answer:` pieces as the model produces them. Use
`/u`, `/d`, and `/h` for transcript paging in the DOS UI. `CHAT\GOLDEN.TXT`
supplies common English call-and-response training examples.
`CHAT\TOKBASE.TXT` is the richer tokenizer basis: it includes dialogue,
response-style sentence pieces, and `CHAT\LEXICON.TSV` grammar words without
forcing all of that scaffolding into the model training stream. The current
CHAT checkpoint is the 4096-token sentence-piece lexicon model tested in QEMU.

Good prompts:

- `Hello, what can you do?`
- `I am bored.`
- `Tell me a joke.`
- `Do you like music?`
- `I want to talk about this DOS demo.`
- `Give me one idea for improving the demo.`
- `What are your limits?`

## DOSHELP

Use `DOSHELP` for FreeDOS, 486 setup, `CONFIG.SYS`, `AUTOEXEC.BAT`, memory,
and batch-file questions. It uses `PACKS\DOSHELP\MODEL` plus retrieval notes
from `DOSHELP\HELP.TXT`.

Good prompts:

- `How do I tune CONFIG.SYS memory?`
- `Write a safe DOS batch file.`
- `What should AUTOEXEC.BAT contain for this demo?`

## OFFICE

Use `OFFICE` for writing tasks: rewrite, summarize, shorten, formalize, and
make prose more professional. It uses `PACKS\OFFICE\MODEL` plus writing notes
from `OFFICE\HELP.TXT`.

Good prompts:

- `Rewrite this memo in a professional tone.`
- `Summarize this paragraph.`
- `Make this note shorter and clearer.`
