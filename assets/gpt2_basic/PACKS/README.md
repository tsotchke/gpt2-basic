# GPT2-BASIC Assistant Packs

Packs are selected inside `ASSIST.EXE` with `/pack NAME`. Type `/packs` to
list them and `/about` to read the current pack's `USAGE.TXT` instructions
inside DOS.

Each pack has human-editable `HELP.TXT` and `KNOW.TXT` files, a generated
`KDB.TXT` recall database, generated `KDBIDX.TXT`/`KDB?.TXT` bucket sidecars,
and a local `USER.TXT` override file. Edit
`HELP.TXT` or `KNOW.TXT` on the host, then run:

```sh
python3 scripts/build_assistant_kdb.py --write
python3 scripts/validate_assistant_pack_authoring.py
```

On DOS, users can edit `USER.TXT` directly for local notes without rebuilding
the pack. Matching `USER.TXT` rows get a retrieval bonus so site-local facts
can override bundled notes. The DOS shell uses the bucket sidecars as a fast
path and falls back to full `KDB.TXT` scan when needed.

## CHAT

Use `CHAT` for ordinary conversation. It is first in `PACKS.TXT`, so the
interactive QEMU demo starts here. It uses `PACKS\CHAT\MODEL` and short
conversation notes from `CHAT\KDB.TXT`/`CHAT\KDB?.TXT` plus local overrides
from `CHAT\USER.TXT`. `KDB.TXT` and bucket sidecars are generated from
`CHAT\HELP.TXT` and `CHAT\KNOW.TXT`. The DOS shell prints `Thinking:`
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
- `How can I ask better questions?`

## DOSHELP

Use `DOSHELP` for FreeDOS, 486 setup, `CONFIG.SYS`, `AUTOEXEC.BAT`, memory,
and batch-file questions. It uses `PACKS\DOSHELP\MODEL` plus retrieval notes
from `DOSHELP\KDB.TXT` and local overrides from `DOSHELP\USER.TXT`.

Good prompts:

- `How do I tune CONFIG.SYS memory?`
- `Write a safe DOS batch file.`
- `What should AUTOEXEC.BAT contain for this demo?`
- `Why use 8.3 filenames in batch files?`

## OFFICE

Use `OFFICE` for writing tasks: rewrite, summarize, shorten, formalize, and
make prose more professional. It uses `PACKS\OFFICE\MODEL` plus writing notes
from `OFFICE\KDB.TXT` and local overrides from `OFFICE\USER.TXT`.

Good prompts:

- `Rewrite this memo in a professional tone.`
- `Summarize this paragraph.`
- `Make this note shorter and clearer.`
- `What belongs in a bug report?`

## DEV

Use `DEV` for software engineering, release work, debugging, testing, and
GPT2-BASIC architecture questions. It shares `PACKS\CHAT\MODEL` but carries
its own `DEV\KDB.TXT` recall database, so it behaves like a lightweight
language cartridge without another model checkpoint.

Good prompts:

- `How should I debug this?`
- `What should I check before release?`
- `How can this feel modern on a 486?`
- `What is retrieval first?`
