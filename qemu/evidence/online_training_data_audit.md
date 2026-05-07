# GPT2-BASIC Online Training Data Audit

This report tracks the online corpus used for the next training pass. The default source set is conservative: Project Gutenberg public-domain text plus NIST/NASA public-government/open-data prose. ShareAlike/GFDL-style sources are opt-in and must stay provenance-tracked.

## Corpus Output

- Combined corpus: `data/online_corpus/online_training_corpus.txt`
- Sources fetched: 6/6
- Clean characters: 837167
- ShareAlike enabled: no

## Source Matrix

| Status | Tier | Source | License posture | Clean chars | Notes |
|---|---|---|---|---:|---|
| `ok` | `permissive` | [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) | [Project Gutenberg public-domain text in the United States after stripping PG boilerplate](https://www.gutenberg.org/policy/license.html) | 126790 | Compact public-domain fiction for dialogue, sentence rhythm, and narrative continuity. |
| `ok` | `permissive` | [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661) | [Project Gutenberg public-domain text in the United States after stripping PG boilerplate](https://www.gutenberg.org/policy/license.html) | 259390 | Readable public-domain prose with direct problem/answer structure. |
| `ok` | `permissive` | [Frankenstein](https://www.gutenberg.org/ebooks/84) | [Project Gutenberg public-domain text in the United States after stripping PG boilerplate](https://www.gutenberg.org/policy/license.html) | 219252 | Long-form explanatory and reflective English, capped for tiny-model training. |
| `ok` | `permissive` | [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) | [Project Gutenberg public-domain text in the United States after stripping PG boilerplate](https://www.gutenberg.org/policy/license.html) | 219655 | Clean public-domain conversational prose, capped to avoid overpowering domain text. |
| `ok` | `permissive` | [NIST SP 800-series general information](https://www.nist.gov/itl/publications-0/nist-special-publication-800-series-general-information) | [NIST SP 800 publications are not subject to U.S. copyright](https://www.nist.gov/itl/publications-0/nist-special-publication-800-series-general-information) | 3529 | Modern technical/government prose for precise explanatory style. |
| `ok` | `permissive` | [NASA Earthdata data use and citation guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) | [NASA-led mission data are CC0 unless otherwise marked; NASA materials generally not copyrighted in the U.S.](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance) | 8551 | Concise policy and technical explanatory prose with explicit provenance guidance. |

## Excluded By Default

| Source | Reason | Reference |
|---|---|---|
| OpenStax computer science and data science books | Current book pages state that the book may not be used for LLM training or ingested into generative AI offerings without permission. | https://openstax.org/books/introduction-computer-science/pages/preface |
| Stack Exchange data dump | Useful technical Q&A, but CC BY-SA versioning and attribution obligations make it unsuitable for the default production corpus. | https://stackoverflow.com/help/licensing |
| Simple English Wikipedia dump | Good small-encyclopedia candidate, but Wikimedia text is GFDL/CC BY-SA; keep behind a later explicit ShareAlike import path. | https://dumps.wikimedia.org/legal.html |
| RFC Editor text corpus | RFCs are freely reproducible unmodified, but training cleanup creates a transformed corpus; use only after legal policy is explicit. | https://www.rfc-editor.org/faq/ |
| FreeBASIC wiki manual | Highly relevant to the runtime, but the official license page clearly covers compiler/runtime licensing rather than granting a clean license for all wiki manual text. | https://www.freebasic.net/wiki/wikka.php?wakka=GnuLicenses |
| OpenWebText, Common Crawl, and random web scrapes | Large and attractive for perplexity, but provenance and redistribution posture are too ambiguous for this production baseline. | https://commoncrawl.org/ |

## Pretraining Command

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --include-docs --corpus-file data/online_corpus/online_training_corpus.txt --corpus-weight 1 --device mps --steps 2500 --output assets/gpt2_basic/MODEL_ONLINE_PRETRAIN
```

Use this as a pretraining base. Fine-tune a candidate on project/runtime domain text, then run host held-out quality before spending QEMU time. Promote to DOS only after host held-out quality beats the active baseline.

## First Candidate Result

`MODEL_ONLINE_CANDIDATE` was trained for 2500 MPS steps from this corpus. It
exported correctly, but host held-out quality was `NEEDS_TRAINING`, 0/5,
average 0.650, which is below the active DOS baseline average of 0.685.

Conclusion: this corpus is useful as a pretraining base, not as the final
production training mix. See
`qemu/evidence/online_training_candidate_report.md`.
